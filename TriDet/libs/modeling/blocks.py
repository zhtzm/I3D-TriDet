import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
# from libs.modeling.weight_init import trunc_normal_


class MaskedConv1D(nn.Module):
    """
    Masked 1D convolution. Interface remains the same as Conv1d.
    Only support a sub set of 1d convs
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode='zeros'
    ):
        super().__init__()
        # 这里强制要求卷积核尺寸为奇数,填充为卷积核尺寸整除2,若步幅为1则尺寸不变
        assert (kernel_size % 2 == 1) and (kernel_size // 2 == padding)
        # 初始化步幅
        self.stride = stride
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode)
        # 初始化偏置,感觉没啥用,大部分时候都会因为使用归一化层而使用偏置
        if bias:
            torch.nn.init.constant_(self.conv.bias, 0.)

    def forward(self, x, mask):
        """
            input: (B, C, T), (B, 1, T)
            outpput: (B, out_channels, (T - 1) // stride + 1), (B, 1, T)
        """
        # x:(B, C, T)
        # mask: (B, 1, T) bool类型
        B, C, T = x.size()
        # 这里要求T必须为不服的整数倍
        assert T % self.stride == 0

        # 卷积操作
        out_conv = self.conv(x)
        # 计算mask
        if self.stride > 1:
            # 这一步是避免步幅不为一而导致out_conv与mask尺寸对不上
            out_mask = F.interpolate(
                mask.to(x.dtype),
                size=T // self.stride,
                mode='nearest'
            )
        else:
            # masking out the features
            out_mask = mask.to(x.dtype)
        # 上面再对其mask的同时将dtype转化率,便于下面使用计算

        # detach就是截断反向传播的梯度流,使得变量没有了梯度反向传播
        out_conv = out_conv * out_mask.detach()
        out_mask = out_mask.bool()
        return out_conv, out_mask


class LayerNorm(nn.Module):
    """
        LayerNorm that supports inputs of size B, C, T
    """

    def __init__(
            self,
            num_channels,
            eps=1e-5,
            affine=True,
            device=None,
            dtype=None,
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(
                torch.ones([1, num_channels, 1], **factory_kwargs))
            self.bias = nn.Parameter(
                torch.zeros([1, num_channels, 1], **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        assert x.dim() == 3
        assert x.shape[1] == self.num_channels

        mu = torch.mean(x, dim=1, keepdim=True)
        res_x = x - mu
        sigma = torch.mean(res_x ** 2, dim=1, keepdim=True)
        out = res_x / torch.sqrt(sigma + self.eps)

        if self.affine:
            out *= self.weight
            out += self.bias

        return out


# helper functions for Transformer blocks
def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])   # (n_position, d_hid)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    # 返回(1, d_hid, n_position) 即 (1, C, T)
    return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)


class ConvBlock(nn.Module):
    """
        A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            n_embd,                 # 输入通道数
            kernel_size=3,          # 卷积核大小
            n_ds_stride=1,          # downsampling stride for the current layer
            expansion_factor=2,     # 隐藏层扩展系数
            n_out=None,             # 输出通道数
            act_layer=nn.ReLU,      # 激活函数
    ):
        super().__init__()
        # 下面的代码保证卷积核大小为奇数,填充大小为卷积核整除2,那么若stride为1则尺寸不变
        assert (kernel_size % 2 == 1) and (kernel_size > 1)
        padding = kernel_size // 2
        if n_out is None:
            n_out = n_embd

        width = int(n_embd * expansion_factor)
        # input: (B, C(n_embd), T) output: (B, width, (T - 1) // n_ds_stride + 1)
        self.conv1 = MaskedConv1D(
            n_embd, width, kernel_size, n_ds_stride, padding=padding)
        # input: (B, width, (T - 1) // n_ds_stride + 1) output: (B, n_out, (T - 1) // n_ds_stride + 1)
        self.conv2 = MaskedConv1D(
            width, n_out, kernel_size, 1, padding=padding)

        # attach downsampling conv op
        if n_ds_stride > 1:
            # 1x1 strided conv (same as resnet)
            self.downsample = MaskedConv1D(n_embd, n_out, 1, n_ds_stride)
        else:
            self.downsample = None

        self.act = act_layer()

    def forward(self, x, mask):
        """
            input: (B, C, T), (B, 1, T)
            output: (B, n_out, (T - 1) // n_ds_stride + 1), (B, 1, (T - 1) // n_ds_stride + 1)
        """
        identity = x
        out, out_mask = self.conv1(x, mask)
        out = self.act(out)
        out, out_mask = self.conv2(out, out_mask)

        # 下采样
        if self.downsample is not None:
            identity, _ = self.downsample(x, mask)

        # 残差链接
        out += identity
        out = self.act(out)

        return out, out_mask


class SGPBlock(nn.Module):
    """
        A simple conv block similar to the basic block used in ResNet
    """

    def __init__(
            self,
            n_embd,                 # 输入通道数
            kernel_size=3,          # 卷积核大小
            n_ds_stride=1,          # downsampling stride for the current layer
            k=1.5,                  # k
            group=1,                # group for cnn
            n_out=None,             # 输出通道数
            n_hidden=None,          # 前馈隐藏层通道数
            path_pdrop=0.0,         # drop path rate
            act_layer=nn.GELU,      # 激活函数
            downsample_type='max',
            init_conv_vars=1        # init gaussian variance for the weight
    ):
        super().__init__()
        # must use odd sized kernel
        # assert (kernel_size % 2 == 1) and (kernel_size > 1)
        # padding = kernel_size // 2

        # 初始化参数
        self.kernel_size = kernel_size
        self.stride = n_ds_stride
        if n_out is None:
            n_out = n_embd

        self.ln = LayerNorm(n_embd)
        self.gn = nn.GroupNorm(16, n_embd)

        assert kernel_size % 2 == 1
        # add 1 to avoid have the same size as the instant-level branch
        up_size = round((kernel_size + 1) * k)
        up_size = up_size + 1 if up_size % 2 == 0 else up_size

        # input(B, C, T) output(B, C, T)
        self.psi = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        # input(B, C, T) output(B, C, T)
        self.fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)
        # input(B, C, T) output(B, C, T)
        self.convw = nn.Conv1d(n_embd, n_embd, kernel_size, stride=1, padding=kernel_size // 2, groups=n_embd)
        # input(B, C, T) output(B, C, T)
        self.convkw = nn.Conv1d(n_embd, n_embd, up_size, stride=1, padding=up_size // 2, groups=n_embd)
        # input(B, C, T) output(B, C, T)
        self.global_fc = nn.Conv1d(n_embd, n_embd, 1, stride=1, padding=0, groups=n_embd)

        # input
        if n_ds_stride > 1:
            if downsample_type == 'max':
                kernel_size, stride, padding = \
                    n_ds_stride + 1, n_ds_stride, (n_ds_stride + 1) // 2
                self.downsample = nn.MaxPool1d(
                    kernel_size, stride=stride, padding=padding)
                self.stride = stride
            elif downsample_type == 'avg':
                self.downsample = nn.Sequential(nn.AvgPool1d(n_ds_stride, stride=n_ds_stride, padding=0),
                                                nn.Conv1d(n_embd, n_embd, 1, 1, 0))
                self.stride = n_ds_stride
            else:
                raise NotImplementedError("downsample type error")
        else:
            self.downsample = nn.Identity()
            self.stride = 1

        # two layer mlp
        if n_hidden is None:
            n_hidden = 4 * n_embd  # default
        if n_out is None:
            n_out = n_embd

        # input(B, C, T) output(B, n_out, T)
        self.mlp = nn.Sequential(
            # input(B, C, T) output(B, 4C, T)
            nn.Conv1d(n_embd, n_hidden, 1, groups=group),
            act_layer(),
            # input(B, C, T) output(B, n_out, T)
            nn.Conv1d(n_hidden, n_out, 1, groups=group),
        )

        # drop path
        if path_pdrop > 0.0:
            self.drop_path_out = AffineDropPath(n_embd, drop_prob=path_pdrop)
            self.drop_path_mlp = AffineDropPath(n_out, drop_prob=path_pdrop)
        else:
            self.drop_path_out = nn.Identity()
            self.drop_path_mlp = nn.Identity()

        self.act = act_layer()
        self.reset_params(init_conv_vars=init_conv_vars)

    def reset_params(self, init_conv_vars=0):
        torch.nn.init.normal_(self.psi.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.fc.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.convkw.weight, 0, init_conv_vars)
        torch.nn.init.normal_(self.global_fc.weight, 0, init_conv_vars)
        torch.nn.init.constant_(self.psi.bias, 0)
        torch.nn.init.constant_(self.fc.bias, 0)
        torch.nn.init.constant_(self.convw.bias, 0)
        torch.nn.init.constant_(self.convkw.bias, 0)
        torch.nn.init.constant_(self.global_fc.bias, 0)

    def forward(self, x, mask):
        """
            input: (B, C, T), (B, 1, T)
            当n_ds_stride为1
                output: (B, n_out, T), (B, n_out, T)
            当downsample_type为max,n_ds_stride为奇数
                output: (B, n_out, T // self.stride + 1), (B, n_out, T // self.stride + 1)
            当downsample_type为max,n_ds_stride为偶数
                output: (B, n_out, (T - 1) // self.stride + 1), (B, n_out, (T - 1) // self.stride + 1)
        """
        # X shape: B, C, T
        B, C, T = x.shape
        x = self.downsample(x)      # x(B, C, T // self.stride + 1) 即 (B, C, N)
        out_mask = F.interpolate(
            mask.to(x.dtype),
            size=torch.div(T, self.stride, rounding_mode='trunc'),
            mode='nearest'
        ).detach()

        out = self.ln(x)                # (B, C, N)
        psi = self.psi(out)             # (B, C, N)
        fc = self.fc(out)               # (B, C, N)
        convw = self.convw(out)         # (B, C, N)
        convkw = self.convkw(out)       # (B, C, N)
        phi = torch.relu(self.global_fc(out.mean(dim=-1, keepdim=True)))
        out = fc * phi + (convw + convkw) * psi + out
        out = x * out_mask + self.drop_path_out(out)

        # FFN
        out = out + self.drop_path_mlp(self.mlp(self.gn(out)))  # (B, n_out, N)

        return out, out_mask.bool()


# drop path: from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
class Scale(nn.Module):
    """
        Multiply the output regression range by a learnable constant value
    """

    def __init__(self, init_value=1.0):
        """
        init_value : initial value for the scalar
        """
        super().__init__()
        self.scale = nn.Parameter(
            torch.tensor(init_value, dtype=torch.float32),
            requires_grad=True
        )

    def forward(self, x):
        """
        input -> scale * input
        """
        return x * self.scale


# The follow code is modified from
# https://github.com/facebookresearch/SlowFast/blob/master/slowfast/models/common.py
def drop_path(x, drop_prob=0.0, training=False):
    """
    Stochastic Depth per sample.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
            x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    mask.floor_()  # binarize
    output = x.div(keep_prob) * mask
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class AffineDropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks) with a per channel scaling factor (and zero init)
    See: https://arxiv.org/pdf/2103.17239.pdf
    """

    def __init__(self, num_dim, drop_prob=0.0, init_scale_value=1e-4):
        super().__init__()
        self.scale = nn.Parameter(
            init_scale_value * torch.ones((1, num_dim, 1)),
            requires_grad=True
        )
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(self.scale * x, self.drop_prob, self.training)
