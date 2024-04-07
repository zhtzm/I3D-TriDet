import math
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import MaskedConv1D, Scale, LayerNorm
from .losses import ctr_diou_loss_1d, sigmoid_focal_loss, ctr_giou_loss_1d
from .models import register_meta_arch, make_backbone, make_neck, make_generator
from ..utils import batched_nms


class ClsHead(nn.Module):
    """
    1D Conv heads for classification
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            num_classes,
            prior_prob=0.01,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            empty_cls=[],
            detach_feat=False
    ):
        super().__init__()
        self.act = act_layer()
        self.detach_feat = detach_feat

        # build the head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        # classifier
        self.cls_head = MaskedConv1D(
            feat_dim, num_classes, kernel_size,
            stride=1, padding=kernel_size // 2
        )

        # use prior in model initialization to improve stability
        # this will overwrite other weight init
        bias_value = -(math.log((1 - prior_prob) / prior_prob))
        torch.nn.init.constant_(self.cls_head.conv.bias, bias_value)

        # a quick fix to empty categories:
        # the weights assocaited with these categories will remain unchanged
        # we set their bias to a large negative value to prevent their outputs
        if len(empty_cls) > 0:
            bias_value = -(math.log((1 - 1e-6) / 1e-6))
            for idx in empty_cls:
                torch.nn.init.constant_(self.cls_head.conv.bias[idx], bias_value)

    def forward(self, fpn_feats, fpn_masks):
        """
            input: 
                fpn_feats: [(B, E, T), (B, E, N1), (B, E, N2), (B, E, N3), ...  (B, E, N(arch[2]))]
                fpn_masks: [(B, 1, T), (B, 1, N1), (B, 1, N2), (B, 1, N3), ...  (B, 1, N(arch[2]))]
            output:
                out_logits: [(B, num_classes, T), (B, num_classes, N1), ...  (B, num_classes, N(arch[2]))]
        """
        assert len(fpn_feats) == len(fpn_masks)

        # apply the classifier for each pyramid level
        out_logits = tuple()
        for _, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            if self.detach_feat:
                cur_out = cur_feat.detach()
            else:
                cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_logits, _ = self.cls_head(cur_out, cur_mask)
            out_logits += (cur_logits,)

        # fpn_masks remains the same
        return out_logits


class RegHead(nn.Module):
    """
    Shared 1D Conv heads for regression
    Simlar logic as PtTransformerClsHead with separated implementation for clarity
    """

    def __init__(
            self,
            input_dim,
            feat_dim,
            fpn_levels,
            num_layers=3,
            kernel_size=3,
            act_layer=nn.ReLU,
            with_ln=False,
            num_bins=16
    ):
        super().__init__()
        self.fpn_levels = fpn_levels
        self.act = act_layer()

        # build the conv head
        self.head = nn.ModuleList()
        self.norm = nn.ModuleList()
        for idx in range(num_layers - 1):
            if idx == 0:
                in_dim = input_dim
                out_dim = feat_dim
            else:
                in_dim = feat_dim
                out_dim = feat_dim
            self.head.append(
                MaskedConv1D(
                    in_dim, out_dim, kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    bias=(not with_ln)
                )
            )
            if with_ln:
                self.norm.append(
                    LayerNorm(out_dim)
                )
            else:
                self.norm.append(nn.Identity())

        self.scale = nn.ModuleList()
        for idx in range(fpn_levels):
            self.scale.append(Scale())

        self.offset_head = MaskedConv1D(
            feat_dim, 2 * (num_bins + 1), kernel_size,
            stride=1, padding=kernel_size // 2
        )

    def forward(self, fpn_feats, fpn_masks):
        """
            input: 
                fpn_feats: [(B, E, T), (B, E, N1), (B, E, N2), (B, E, N3), ...  (B, E, N(arch[2]))]
                fpn_masks: [(B, 1, T), (B, 1, N1), (B, 1, N2), (B, 1, N3), ...  (B, 1, N(arch[2]))]
            output:
                out_offsets: [(B, 2 * (num_bins + 1), T), (B, 2 * (num_bins + 1), N1) ...  (B, 2 * (num_bins + 1), N(arch[2]))]
        """
        assert len(fpn_feats) == len(fpn_masks)
        assert len(fpn_feats) == self.fpn_levels

        # apply the classifier for each pyramid level
        out_offsets = tuple()
        for l, (cur_feat, cur_mask) in enumerate(zip(fpn_feats, fpn_masks)):
            cur_out = cur_feat
            for idx in range(len(self.head)):
                cur_out, _ = self.head[idx](cur_out, cur_mask)
                cur_out = self.act(self.norm[idx](cur_out))
            cur_offsets, _ = self.offset_head(cur_out, cur_mask)
            out_offsets += (F.relu(self.scale[l](cur_offsets)),)

        # fpn_masks remains the same
        return out_offsets


@register_meta_arch("TriDet")
class TriDet(nn.Module):
    """
        Transformer based model for single stage action localization
    """

    def __init__(
            self,
            backbone_type,  # a string defines which backbone we use
            fpn_type,  # a string defines which fpn we use
            backbone_arch,  # a tuple defines # layers in embed / stem / branch
            scale_factor,  # scale factor between branch layers
            input_dim,  # input feat dim
            max_seq_len,  # max sequence length (used for training)
            max_buffer_len_factor,  # max buffer size (defined a factor of max_seq_len)
            n_sgp_win_size,  # window size w for sgp
            embd_kernel_size,  # kernel size of the embedding network
            embd_dim,  # output feat channel of the embedding network
            embd_with_ln,  # attach layernorm to embedding network
            fpn_dim,  # feature dim on FPN,
            sgp_mlp_dim,  # the numnber of dim in SGP
            fpn_with_ln,  # if to apply layer norm at the end of fpn
            head_dim,  # feature dim for head
            regression_range,  # regression range on each level of FPN
            head_num_layers,  # number of layers in the head (including the classifier)
            head_kernel_size,  # kernel size for reg/cls heads
            boudary_kernel_size,  # kernel size for boundary heads
            head_with_ln,  # attache layernorm to reg/cls heads
            use_abs_pe,  # if to use abs position encoding
            num_bins,  # the bin number in Trident-head (exclude 0)
            iou_weight_power,  # the power of iou weight in loss
            downsample_type,  # how to downsample feature in FPN
            input_noise,  # add gaussian noise with the variance, play a similar role to position embedding
            k,  # the K in SGP
            init_conv_vars,  # initialization of gaussian variance for the weight in SGP
            use_trident_head,  # if use the Trident-head
            num_classes,  # number of action classes
            train_cfg,  # other cfg for training
            test_cfg  # other cfg for testing
    ):
        super().__init__()
        # re-distribute params to backbone / neck / head
        self.fpn_strides = [scale_factor ** i for i in range(backbone_arch[-1] + 1)]

        self.input_noise = input_noise

        self.reg_range = regression_range
        assert len(self.fpn_strides) == len(self.reg_range)
        self.scale_factor = scale_factor
        self.iou_weight_power = iou_weight_power
        # #classes = num_classes + 1 (background) with last category as background
        # e.g., num_classes = 10 -> 0, 1, ..., 9 as actions, 10 as background
        self.num_classes = num_classes

        # check the feature pyramid and local attention window size
        self.max_seq_len = max_seq_len
        if isinstance(n_sgp_win_size, int):
            self.sgp_win_size = [n_sgp_win_size] * len(self.fpn_strides)
        else:
            assert len(n_sgp_win_size) == len(self.fpn_strides)
            self.sgp_win_size = n_sgp_win_size
        max_div_factor = 1
        for l, (s, w) in enumerate(zip(self.fpn_strides, self.sgp_win_size)):
            stride = s * w if w > 1 else s
            if max_div_factor < stride:
                max_div_factor = stride
        self.max_div_factor = max_div_factor

        # training time config
        self.train_center_sample = train_cfg['center_sample']
        assert self.train_center_sample in ['radius', 'none']
        self.train_center_sample_radius = train_cfg['center_sample_radius']
        self.train_loss_weight = train_cfg['loss_weight']
        self.train_cls_prior_prob = train_cfg['cls_prior_prob']
        self.train_dropout = train_cfg['dropout']
        self.train_droppath = train_cfg['droppath']
        self.train_label_smoothing = train_cfg['label_smoothing']

        # test time config
        self.test_pre_nms_thresh = test_cfg['pre_nms_thresh']
        self.test_pre_nms_topk = test_cfg['pre_nms_topk']
        self.test_iou_threshold = test_cfg['iou_threshold']
        self.test_min_score = test_cfg['min_score']
        self.test_max_seg_num = test_cfg['max_seg_num']
        self.test_nms_method = test_cfg['nms_method']
        assert self.test_nms_method in ['soft', 'hard', 'none']
        self.test_duration_thresh = test_cfg['duration_thresh']
        self.test_multiclass_nms = test_cfg['multiclass_nms']
        self.test_nms_sigma = test_cfg['nms_sigma']
        self.test_voting_thresh = test_cfg['voting_thresh']
        self.num_bins = num_bins
        self.use_trident_head = use_trident_head

        # we will need a better way to dispatch the params to backbones / necks
        # backbone network: conv + transformer
        assert backbone_type in ['SGP', 'conv']
        if backbone_type == 'SGP':
            self.backbone = make_backbone(
                'SGP',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'sgp_mlp_dim': sgp_mlp_dim,
                    'n_embd_ks': embd_kernel_size,
                    'max_len': max_seq_len,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln': embd_with_ln,
                    'path_pdrop': self.train_droppath,
                    'downsample_type': downsample_type,
                    'sgp_win_size': self.sgp_win_size,
                    'use_abs_pe': use_abs_pe,
                    'k': k,
                    'init_conv_vars': init_conv_vars
                }
            )
        else:
            self.backbone = make_backbone(
                'conv',
                **{
                    'n_in': input_dim,
                    'n_embd': embd_dim,
                    'n_embd_ks': embd_kernel_size,
                    'arch': backbone_arch,
                    'scale_factor': scale_factor,
                    'with_ln': embd_with_ln
                }
            )

        # fpn network: convs
        assert fpn_type in ['fpn', 'identity']
        self.neck = make_neck(
            fpn_type,
            **{
                'in_channels': [embd_dim] * (backbone_arch[-1] + 1),
                'out_channel': fpn_dim,
                'scale_factor': scale_factor,
                'with_ln': fpn_with_ln
            }
        )

        # location generator: points
        self.point_generator = make_generator(
            'point',
            **{
                'max_seq_len': max_seq_len * max_buffer_len_factor,
                'fpn_levels': len(self.fpn_strides),
                'scale_factor': scale_factor,
                'regression_range': self.reg_range,
                'strides': self.fpn_strides
            }
        )

        # classfication and regerssion heads
        self.cls_head = ClsHead(
            fpn_dim, head_dim, self.num_classes,
            kernel_size=head_kernel_size,
            prior_prob=self.train_cls_prior_prob,
            with_ln=head_with_ln,
            num_layers=head_num_layers,
            empty_cls=train_cfg['head_empty_cls']
        )

        if use_trident_head:
            self.start_head = ClsHead(
                fpn_dim, head_dim, self.num_classes,
                kernel_size=boudary_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                with_ln=head_with_ln,
                num_layers=head_num_layers,
                empty_cls=train_cfg['head_empty_cls'],
                detach_feat=True
            )
            self.end_head = ClsHead(
                fpn_dim, head_dim, self.num_classes,
                kernel_size=boudary_kernel_size,
                prior_prob=self.train_cls_prior_prob,
                with_ln=head_with_ln,
                num_layers=head_num_layers,
                empty_cls=train_cfg['head_empty_cls'],
                detach_feat=True
            )

            self.reg_head = RegHead(
                fpn_dim, head_dim, len(self.fpn_strides),
                kernel_size=head_kernel_size,
                num_layers=head_num_layers,
                with_ln=head_with_ln,
                num_bins=num_bins
            )
        else:
            self.reg_head = RegHead(
                fpn_dim, head_dim, len(self.fpn_strides),
                kernel_size=head_kernel_size,
                num_layers=head_num_layers,
                with_ln=head_with_ln,
                num_bins=0
            )

        # maintain an EMA of #foreground to stabilize the loss normalizer
        # useful for small mini-batch training
        self.loss_normalizer = train_cfg['init_loss_norm']
        self.loss_normalizer_momentum = 0.9

    @property
    def device(self):
        # a hacky way to get the device type
        # will throw an error if parameters are on different devices
        return list(set(p.device for p in self.parameters()))[0]

    def decode_offset(self, out_offsets, pred_start_neighbours, pred_end_neighbours):
        """
            use_trident_head:
            traininput:
                out_offsets: [(B, T, 2 * (num_bins + 1)), (B, N1, 2 * (num_bins + 1)) ...  (B, N(arch[2]), 2 * (num_bins + 1))]
                pred_start_neighbours: [(B, T, num_classes, num_bins + 1), (B, N1, num_classes, num_bins + 1) ... (B, N(arch[2]), num_classes, num_bins + 1)]
                pred_end_neighbours: [(B, T, num_classes, num_bins + 1), (B, N1, num_classes, num_bins + 1) ... (B, N(arch[2]), num_classes, num_bins + 1)]
            inferenceinput:
                out_offsets: [(T, 2 * (num_bins + 1)), (N1, 2 * (num_bins + 1)) ...  (N(arch[2]), 2 * (num_bins + 1))]
                pred_start_neighbours: [(T, num_classes, num_bins + 1), (N1, num_classes, num_bins + 1) ... (N(arch[2]), num_classes, num_bins + 1)]
                pred_end_neighbours: [(T, num_classes, num_bins + 1), (N1, num_classes, num_bins + 1) ... (N(arch[2]), num_classes, num_bins + 1)]
            trainoutput:
                decode_offset: (B, S, num_classes, 1)
            inferenceoutput:
                decode_offset: (S, num_classes, 1)
        """

        if not self.use_trident_head:
            # 如果不是trident_head简单处理一下返回
            if self.training:
                out_offsets = torch.cat(out_offsets, dim=1)
            return out_offsets
        else:
            # Make an adaption for train and validation, when training, the out_offsets is a list with feature outputs
            # from each FPN level. Each feature with shape [batchsize, T_level, (Num_bin+1)x2].
            # For validation, the out_offsets is a feature with shape [T_level, (Num_bin+1)x2]
            if self.training:
                # (B, S, 2 * (num_bins + 1))
                out_offsets = torch.cat(out_offsets, dim=1)
                # (B, S, num_classes, num_bins + 1)
                pred_start_neighbours = torch.cat(pred_start_neighbours, dim=1)
                # (B, S, num_classes, num_bins + 1)
                pred_end_neighbours = torch.cat(pred_end_neighbours, dim=1)

                # (B, S, 2, num_bins + 1)
                out_offsets = out_offsets.view(out_offsets.shape[:2] + (2, -1))
                # 这里就是利用广播机制pred_start_neighbours + out_offsets[:, :, 0, :],再对num_bins + 1进行softmax   (B, S, num_classes, num_bins + 1)
                pred_left_dis = torch.softmax(pred_start_neighbours + out_offsets[:, :, :1, :], dim=-1)
                # 这里就是利用广播机制pred_end_neighbours + out_offsets[:, :, 1, :],再对num_bins + 1进行softmax     (B, S, num_classes, num_bins + 1)
                pred_right_dis = torch.softmax(pred_end_neighbours + out_offsets[:, :, 1:, :], dim=-1)
            else:
                # (S, 2, num_bins + 1)
                out_offsets = out_offsets.view(out_offsets.shape[0], 2, -1)
                # 这里就是利用广播机制pred_start_neighbours + out_offsets[:, :, 0, :],再对num_bins + 1进行softmax   (S, num_classes, num_bins + 1)
                pred_left_dis = torch.softmax(pred_start_neighbours + out_offsets[None, :, 0, :], dim=-1)
                # 这里就是利用广播机制pred_end_neighbours + out_offsets[:, :, 1, :],再对num_bins + 1进行softmax     (S, num_classes, num_bins + 1)
                pred_right_dis = torch.softmax(pred_end_neighbours + out_offsets[None, :, 1, :], dim=-1)


            max_range_num = pred_left_dis.shape[-1]
            # left_range_idx(max_range_num, 1) 即 (num_bins + 1, 1)
            left_range_idx = torch.arange(max_range_num - 1, -1, -1, device=pred_start_neighbours.device,
                                          dtype=torch.float).unsqueeze(-1)
            # right_range_idx(max_range_num, 1) 即 (num_bins + 1, 1)
            right_range_idx = torch.arange(max_range_num, device=pred_end_neighbours.device,
                                           dtype=torch.float).unsqueeze(-1)

            # TODO
            # 原来:pred_left_dis = pred_left_dis.masked_fill(torch.isnan(pred_right_dis), 0)
            # 疑问为什么pred_right_dis
            # 去除nan值
            pred_left_dis = pred_left_dis.masked_fill(torch.isnan(pred_left_dis), 0)
            # 去除nan值
            pred_right_dis = pred_right_dis.masked_fill(torch.isnan(pred_right_dis), 0)

            # calculate the value of expectation for the offset:
            # (B, S, num_classes, num_bins + 1) x (num_bins + 1, 1) 求期望 = (B, S, num_classes, 1)
            decoded_offset_left = torch.matmul(pred_left_dis, left_range_idx)
            # (B, S, num_classes, num_bins + 1) x (num_bins + 1, 1) 求期望 = (B, S, num_classes, 1)
            decoded_offset_right = torch.matmul(pred_right_dis, right_range_idx)
            return torch.cat([decoded_offset_left, decoded_offset_right], dim=-1)

    def forward(self, video_list):
        # 预处理得到inputs(B, C, T) mask(B, 1, T)
        batched_inputs, batched_masks = self.preprocessing(video_list)

        # forward the network (backbone -> neck -> heads)
        """
            feats: [(B, E, T), (B, E, N1), (B, E, N2), (B, E, N3), ...  (B, E, N(arch[2]))]
            masks: [(B, 1, T), (B, 1, N1), (B, 1, N2), (B, 1, N3), ...  (B, 1, N(arch[2]))]
        """
        feats, masks = self.backbone(batched_inputs, batched_masks)
        """
            因为参数设置都使用start_level, end_level的默认值,因此这里仍然与原来一致
            fpn_feats: [(B, E, T), (B, E, N1), (B, E, N2), (B, E, N3), ...  (B, E, N(arch[2]))]
            fpn_masks: [(B, 1, T), (B, 1, N1), (B, 1, N2), (B, 1, N3), ...  (B, 1, N(arch[2]))]
        """
        fpn_feats, fpn_masks = self.neck(feats, masks)

        # points: [(T, 4), (N1, 4), (N2, 4), (N3, 4), ...  (N(arch[2]), 4)]
        points = self.point_generator(fpn_feats)

        """
            这里将每个时刻进行分类
            out_cls_logits: [(B, num_classes, T), (B, num_classes, N1), ...  (B, num_classes, N(arch[2]))]
        """
        out_cls_logits = self.cls_head(fpn_feats, fpn_masks)

        if self.use_trident_head:
            """
                这里将每个时刻进行分类
                out_lb_logits: [(B, num_classes, T), (B, num_classes, N1), ...  (B, num_classes, N(arch[2]))]
            """
            out_lb_logits = self.start_head(fpn_feats, fpn_masks)
            """
                这里将每个时刻进行分类
                out_rb_logits: [(B, num_classes, T), (B, num_classes, N1), ...  (B, num_classes, N(arch[2]))]
            """
            out_rb_logits = self.end_head(fpn_feats, fpn_masks)
        else:
            out_lb_logits = None
            out_rb_logits = None

        """
            out_offsets: [(B, 2 * (num_bins + 1), T), (B, 2 * (num_bins + 1), N1) ...  (B, 2 * (num_bins + 1), N(arch[2]))]
        """
        out_offsets = self.reg_head(fpn_feats, fpn_masks)

        # 变换纬度
        # out_cls_logits: [(B, num_classes, T), (B, num_classes, N1), ...  (B, num_classes, N(arch[2]))] 
        # -> [(B, T, num_classes), (B, N1, num_classes), ...  (B, N(arch[2]), num_classes)] 
        out_cls_logits = [x.permute(0, 2, 1) for x in out_cls_logits]
        # out_offset: [(B, 2 * (num_bins + 1), T), (B, 2 * (num_bins + 1), N1) ...  (B, 2 * (num_bins + 1), N(arch[2]))]
        # -> [(B, T, 2 * (num_bins + 1)), (B, N1, 2 * (num_bins + 1)) ...  (B, N(arch[2]), 2 * (num_bins + 1))]
        out_offsets = [x.permute(0, 2, 1) for x in out_offsets]
        # fpn_masks: [(B, 1, T), (B, 1, N1), (B, 1, N2), (B, 1, N3), ...  (B, 1, N(arch[2]))]
        # -> [(B, T), (B, N1), (B, N2), (B, N3), ...  (B, N(arch[2]))]
        fpn_masks = [x.squeeze(1) for x in fpn_masks]

        # return loss during training
        if self.training:
            # generate segment/lable List[N x 2] / List[N] with length = B
            assert video_list[0]['segments'] is not None, "GT action labels does not exist"
            assert video_list[0]['labels'] is not None, "GT action labels does not exist"
            # (B, N, 2)
            gt_segments = [x['segments'].to(self.device) for x in video_list]
            # (B, N)
            gt_labels = [x['labels'].to(self.device) for x in video_list]

            # compute the gt labels for cls & reg
            # list of prediction targets
            gt_cls_labels, gt_offsets = self.label_points(
                points, gt_segments, gt_labels)

            # compute the loss and return
            losses = self.losses(
                fpn_masks,
                out_cls_logits, out_offsets,
                gt_cls_labels, gt_offsets,
                out_lb_logits, out_rb_logits,
            )
            return losses

        else:
            # decode the actions (sigmoid / stride, etc)
            results = self.inference(
                video_list, points, fpn_masks,
                out_cls_logits, out_offsets,
                out_lb_logits, out_rb_logits,
            )
            return results

    @torch.no_grad()
    def preprocessing(self, video_list, padding_val=0.0):
        """
            Generate batched features and masks from a list of dict items
        """
        feats = [x['feats'] for x in video_list]
        feats_lens = torch.as_tensor([feat.shape[-1] for feat in feats])
        max_len = feats_lens.max(0).values.item()

        if self.training:
            assert max_len <= self.max_seq_len, "Input length must be smaller than max_seq_len during training"
            # set max_len to self.max_seq_len
            max_len = self.max_seq_len
            # batch input shape B, C, T
            batch_shape = [len(feats), feats[0].shape[0], max_len]
            batched_inputs = feats[0].new_full(batch_shape, padding_val)
            for feat, pad_feat in zip(feats, batched_inputs):
                pad_feat[..., :feat.shape[-1]].copy_(feat)
            if self.input_noise > 0:
                # trick, adding noise slightly increases the variability between input features.
                noise = torch.randn_like(batched_inputs) * self.input_noise
                batched_inputs += noise
        else:
            assert len(video_list) == 1, "Only support batch_size = 1 during inference"
            # input length < self.max_seq_len, pad to max_seq_len
            if max_len <= self.max_seq_len:
                max_len = self.max_seq_len
            else:
                # pad the input to the next divisible size
                stride = self.max_div_factor
                max_len = (max_len + (stride - 1)) // stride * stride
            padding_size = [0, max_len - feats_lens[0]]
            batched_inputs = F.pad(feats[0], padding_size, value=padding_val).unsqueeze(0)

        # generate the mask
        batched_masks = torch.arange(max_len)[None, :] < feats_lens[:, None]

        # push to device
        batched_inputs = batched_inputs.to(self.device)
        batched_masks = batched_masks.unsqueeze(1).to(self.device)

        return batched_inputs, batched_masks

    @torch.no_grad()
    def label_points(self, points, gt_segments, gt_labels):
        """
            input:
                points: [(T, 4), (N1, 4), (N2, 4), (N3, 4), ...  (N(arch[2]), 4)]
                gt_segments: (B, N, 2)
                gt_labels: (B, N)
            output:
                gt_cls: (B, S, num_classes)
                gt_offset: (B, S, 2)
        """
        num_levels = len(points)
        # concat_points (Sum, 4) 即 (S, 4) (ts, reg_range[0], reg_range[0], stride)
        concat_points = torch.cat(points, dim=0)
        gt_cls, gt_offset = [], []

        # loop over each video sample
        for gt_segment, gt_label in zip(gt_segments, gt_labels):
            cls_targets, reg_targets = self.label_points_single_video(
                concat_points, gt_segment, gt_label
            )
            # append to list (len = # images, each of size FT x C)
            gt_cls.append(cls_targets)
            gt_offset.append(reg_targets)

        return gt_cls, gt_offset

    @torch.no_grad()
    def label_points_single_video(self, concat_points, gt_segment, gt_label):
        """
            input:
                concat_points: (S, 4)
                gt_segment: (N, 2)
                gt_label: (N)
            output:
                cls_targets: (S, num_classes)
                reg_targets: (S, 2)
        """
        num_pts = concat_points.shape[0]
        num_gts = gt_segment.shape[0]

        # corner case where current sample does not have actions
        if num_gts == 0:
            cls_targets = gt_segment.new_full((num_pts, self.num_classes), 0)
            reg_targets = gt_segment.new_zeros((num_pts, 2))
            return cls_targets, reg_targets

        # lens (N)
        lens = gt_segment[:, 1] - gt_segment[:, 0]
        # lens (S, N)
        lens = lens[None, :].repeat(num_pts, 1)

        # gt_segs (S, N, 2)
        gt_segs = gt_segment[None].expand(num_pts, num_gts, 2)
        # left: (S, N)
        left = concat_points[:, 0, None] - gt_segs[:, :, 0]
        # right: (S, N)
        right = gt_segs[:, :, 1] - concat_points[:, 0, None]
        # reg_targets (S, N, 2)
        reg_targets = torch.stack((left, right), dim=-1)

        if self.train_center_sample == 'radius':
            # center_pts (S, N) 每个元素代表gt视频段的中心点
            center_pts = 0.5 * (gt_segs[:, :, 0] + gt_segs[:, :, 1])
            """
                根据gt中心点与不同range的上限计算最小边界与最大边界
            """
            t_mins = \
                center_pts - concat_points[:, 3, None] * self.train_center_sample_radius    # (S, N)
            t_maxs = \
                center_pts + concat_points[:, 3, None] * self.train_center_sample_radius    # (S, N)
            # prevent t_mins / maxs from over-running the action boundary
            # left: torch.maximum(t_mins, gt_segs[:, :, 0])
            # right: torch.minimum(t_maxs, gt_segs[:, :, 1])
            # F T x N (distance to the new boundary)
            cb_dist_left = concat_points[:, 0, None] - torch.maximum(t_mins, gt_segs[:, :, 0])
            cb_dist_right = torch.minimum(t_maxs, gt_segs[:, :, 1]) - concat_points[:, 0, None]
            # F T x N x 2
            center_seg = torch.stack((cb_dist_left, cb_dist_right), -1)
            # 计算在gt视频段中的点 inside_gt_seg_mask(S, N)
            inside_gt_seg_mask = center_seg.min(-1)[0] > 0

        else:
            # 计算,在gt视频段中的点 inside_gt_seg_mask(S, N)
            inside_gt_seg_mask = reg_targets.min(-1)[0] > 0

        # 限制左右边界在range之内 max_regress_distance(S, N)
        max_regress_distance = reg_targets.max(-1)[0]
        # (S, N)
        inside_regress_range = torch.logical_and(
            (max_regress_distance >= concat_points[:, 1, None]),
            (max_regress_distance <= concat_points[:, 2, None])
        )

        # 接下来两部是将lens (S, N),进行mask
        # 对每一个视频点s,在它满足两个限制的n上为它原来的值,否则为float('inf')
        lens.masked_fill_(inside_gt_seg_mask == 0, float('inf'))
        lens.masked_fill_(inside_regress_range == 0, float('inf'))
        # 这一步挑出了每一个s中的最小的n(gt),并得到相应下标
        # min_len(S) min_len_inds(S)
        min_len, min_len_inds = lens.min(dim=1)

        # 这里是对每一个视频点s唯一配对n(gt),用来应对及其相似的两个视频段
        min_len_mask = torch.logical_and(
            (lens <= (min_len[:, None] + 1e-3)), (lens < float('inf'))
        ).to(reg_targets.dtype)

        # 这里对视频标签进行独热编码 gt_label_one_hot(N, num_classes)
        gt_label_one_hot = F.one_hot(
            gt_label, self.num_classes
        ).to(reg_targets.dtype)
        # min_len_mask(S, N) gt_label_one_hot(N, num_classes)
        # cls_targets(S, num_classes) 这是对匹配到n的s进行类别标注,匹配对应片段类别的独热编码
        cls_targets = min_len_mask @ gt_label_one_hot   
        # to prevent multiple GT actions with the same label and boundaries
        cls_targets.clamp_(min=0.0, max=1.0)
        # reg_targets (S, N, 2) -> reg_targets (S, 2)
        # 这里使用min_len_inds,挑选出对应的视频段偏移量
        reg_targets = reg_targets[range(num_pts), min_len_inds]
        # normalization based on stride,其实就是计算偏移百分比,归为0~1
        reg_targets /= concat_points[:, 3, None]

        return cls_targets, reg_targets

    def losses(
            self, fpn_masks,
            out_cls_logits, out_offsets,
            gt_cls_labels, gt_offsets,
            out_start, out_end,
    ):
        """
            input:
                fpn_masks: [(B, T), (B, N1), (B, N2), (B, N3), ...  (B, N(arch[2]))]
                out_cls_logits: [(B, T, num_classes), (B, N1, num_classes), ...  (B, N(arch[2]), num_classes)] 
                out_offsets: [(B, T, 2 * (num_bins + 1)), (B, N1, 2 * (num_bins + 1)) ...  (B, N(arch[2]), 2 * (num_bins + 1))]
                gt_cls_labels: (B, S, num_classes)
                gt_offsets: (B, S, 2)
                out_start: [(B, num_classes, T), (B, num_classes, N1), ...  (B, num_classes, N(arch[2]))]
                out_end: [(B, num_classes, T), (B, num_classes, N1), ...  (B, num_classes, N(arch[2]))]
        """
        valid_mask = torch.cat(fpn_masks, dim=1)    #(B, S)

        if self.use_trident_head:
            out_start_logits = []
            out_end_logits = []
            for i in range(len(out_start)):
                """
                    下面pad left的目的是对于一个out_start[i](B, num_classes, T)进行变换,得到(B, T, num_class, num_bins + 1)
                    含义是每个b的每一时间点,不同类别的该点以及前num_bins的特征,若前面没有就用0来填充
                """
                # x (B, num_classes, T + num_bins, 1),每一个类下面的(T + num_bins)前num_bins是0,便于下面得到最终特征的前面元素再前面无元素的填充
                x = (F.pad(out_start[i], (self.num_bins, 0), mode='constant', value=0)).unsqueeze(-1) # pad left
                # 得到x的形状列表 (B, num_classes, T + num_bins, 1)
                x_size = list(x.size())  
                # 最后一维为num_bins + 1就是我们想要的该点以及前num_bins
                x_size[-1] = self.num_bins + 1  
                # 下面这个相减就是将第三维回复, 操作完x_size为(B, num_classes, T, num_bins + 1)
                x_size[-2] = x_size[-2] - self.num_bins 
                # 得到原来的x_stride,这个简单来说就是x中元素是怎么排列的
                x_stride = list(x.stride())
                # 这里将x_stride[-2]赋值x_stride[-1],其实本质就是修改为1,因为x (B, num_classes, T + num_bins, 1)最后一维本就是按1排列的
                x_stride[-2] = x_stride[-1]

                # 这里修改让x中值按x_stride排列到x_size中
                """
                    x (B, num_classes, T + num_bins, 1)
                    x_stride = (num_classes * T + num_bins * 1, T + num_bins * 1, 1, 1)
                    x_size = (B, num_classes, T, num_bins + 1)
                    这里可以使用一下jupyter notebook写一个例子
                    可以看出这样排列就是希望的效果  
                """
                x = x.as_strided(size=x_size, stride=x_stride)
                # (B, T, num_classes, num_bins + 1)
                out_start_logits.append(x.permute(0, 2, 1, 3))

                """
                    下面pad left的目的是对于一个out_start[i](B, num_classes, T)进行变换,得到(B, T, num_class, num_bins + 1)
                    含义是每个b的每一时间点,不同类别的该点以及后num_bins的特征,若后面没有就用0来填充
                """
                x = (F.pad(out_end[i], (0, self.num_bins), mode='constant', value=0)).unsqueeze(-1)  # pad right
                x = x.as_strided(size=x_size, stride=x_stride)
                # (B, T, num_classes, num_bins + 1)
                out_end_logits.append(x.permute(0, 2, 1, 3))
        else:
            out_start_logits = None
            out_end_logits = None

        # 1. classification loss
        # 这一步就是简单的将gt_cls_labels转换为Tensor格式,本质上gt_cls就是gt_cls_labels
        gt_cls = torch.stack(gt_cls_labels)
        # gt_cls(B, S, num_classes),最后一维是类别的独热编码,在之前的处理中,仔细观察可以知道当这一维全零代表没有匹配和类别,下面就是的得到是否是有效视频点的mask
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)  # (B, S)

        decoded_offsets = self.decode_offset(out_offsets, out_start_logits, out_end_logits)  # (B, S, num_classes, 2)
        decoded_offsets = decoded_offsets[pos_mask] # 这一步将decoded_offsets中有效视频点的decoded_offsets挑出来(M, num_classes, 2)

        if self.use_trident_head:
            # 这里将gt_cls中有效视频点的gt_cls挑出来并将其转化为bool型,(M, num_classes)
            gt_cls_pos_mask = gt_cls[pos_mask].bool()
            # 将decoded_offsets进一步筛选,只留有对应类别的decoded_offsets,(M, 2)
            pred_offsets = decoded_offsets[gt_cls_pos_mask]
            # (M)
            vid = torch.where(gt_cls[pos_mask])[0]
            # (B, S, 2) -> (M, 2)
            tmp = torch.stack(gt_offsets)[pos_mask]
            gt_offsets = tmp[vid]   # (M, 2)
        else:
            pred_offsets = decoded_offsets
            gt_offsets = torch.stack(gt_offsets)[pos_mask]

        # 其实就是M
        # 正样本数量num_pos通过对pos_mask求和得到，这里pos_mask标记了所有正样本的位置。
        num_pos = pos_mask.sum().item()
        # loss_normalizer通过结合动量（loss_normalizer_momentum）和当前批次中的正样本数量（至少为1）动态更新。
        # 这种方法有助于稳定训练过程，尤其是在每个批次中正样本数量波动较大时。
        self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
        ) * max(num_pos, 1)

        # gt_cls(B, S, num_classes) valid_mask(B, S) -> (M_ex, num_classes) M是有效视频点,M_ex是可用视频点
        # gt_target是通过应用valid_mask到gt_cls上获得的，这样只选取了有效的类别标签
        gt_target = gt_cls[valid_mask]

        # optinal label smoothing
        gt_target *= 1 - self.train_label_smoothing
        gt_target += self.train_label_smoothing / (self.num_classes + 1)

        # 计算focal Loss
        # 使用sigmoid_focal_loss计算分类损失，这种损失函数特别适合处理类别不平衡的问题，因为它可以减少对易分类样本的关注，而增加对难分类样本的关注。
        # 损失是在通过valid_mask筛选后的输出分类概率和经过标签平滑处理的gt_target之间计算的
        # gt_target(M_ex, num_classes)
        # torch.cat(out_cls_logits, dim=1)[valid_mask] (M_ex, num_classes)
        cls_loss = sigmoid_focal_loss(
            torch.cat(out_cls_logits, dim=1)[valid_mask],
            gt_target,
            reduction='none'
        )

        if self.use_trident_head:
            # couple the classification loss with iou score
            iou_rate = ctr_giou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='none'
            )
            rated_mask = gt_target > self.train_label_smoothing / (self.num_classes + 1)
            cls_loss[rated_mask] *= (1 - iou_rate) ** self.iou_weight_power

        # 将计算出的分类损失求和，并通过之前动态更新的loss_normalizer进行归一化。
        # 这个归一化步骤有助于保持损失在一个合理的范围内，确保不同批次间的损失是可比的。
        cls_loss = cls_loss.sum()
        cls_loss /= self.loss_normalizer

        # 2. regression using IoU/GIoU loss (defined on positive samples)
        if num_pos == 0:
            reg_loss = 0 * pred_offsets.sum()
        else:
            # giou loss defined on positive samples
            reg_loss = ctr_diou_loss_1d(
                pred_offsets,
                gt_offsets,
                reduction='sum'
            )
            reg_loss /= self.loss_normalizer

        if self.train_loss_weight > 0:
            loss_weight = self.train_loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        # return a dict of losses
        final_loss = cls_loss + reg_loss * loss_weight
        return {'cls_loss': cls_loss,
                'reg_loss': reg_loss,
                'final_loss': final_loss}

    @torch.no_grad()
    def inference(
            self,
            video_list,
            points, fpn_masks,
            out_cls_logits, out_offsets,
            out_lb_logits, out_rb_logits,
    ):
        """
            video_list: (B)
            points: [(T, 4), (N1, 4), (N2, 4), (N3, 4), ...  (N(arch[2]), 4)]
            fpn_masks: [(B, T), (B, N1), (B, N2), (B, N3), ...  (B, N(arch[2]))]
            out_cls_logits: [(B, T, num_classes), (B, N1, num_classes), ...  (B, N(arch[2]), num_classes)] 
            out_offsets: [(B, T, 2 * (num_bins + 1)), (B, N1, 2 * (num_bins + 1)) ...  (B, N(arch[2]), 2 * (num_bins + 1))]
            out_lb_logits: [(B, num_classes, T), (B, num_classes, N1), ...  (B, num_classes, N(arch[2]))]
            out_rb_logits: [(B, num_classes, T), (B, num_classes, N1), ...  (B, num_classes, N(arch[2]))]
        """
        results = []

        # 1: gather video meta information
        vid_idxs = [x['video_id'] for x in video_list]
        vid_fps = [x['fps'] for x in video_list]
        vid_lens = [x['duration'] for x in video_list]
        vid_ft_stride = [x['feat_stride'] for x in video_list]
        vid_ft_nframes = [x['feat_num_frames'] for x in video_list]

        # 2: inference on each single video and gather the results
        # upto this point, all results use timestamps defined on feature grids
        for idx, (vidx, fps, vlen, stride, nframes) in enumerate(
                zip(vid_idxs, vid_fps, vid_lens, vid_ft_stride, vid_ft_nframes)
        ):
            # gather per-video outputs
            cls_logits_per_vid = [x[idx] for x in out_cls_logits]
            offsets_per_vid = [x[idx] for x in out_offsets]
            fpn_masks_per_vid = [x[idx] for x in fpn_masks]

            if self.use_trident_head:
                lb_logits_per_vid = [x[idx] for x in out_lb_logits]
                rb_logits_per_vid = [x[idx] for x in out_rb_logits]
            else:
                lb_logits_per_vid = [None for x in range(len(out_cls_logits))]
                rb_logits_per_vid = [None for x in range(len(out_cls_logits))]

            # inference on a single video (should always be the case)
            results_per_vid = self.inference_single_video(
                points, fpn_masks_per_vid,
                cls_logits_per_vid, offsets_per_vid,
                lb_logits_per_vid, rb_logits_per_vid
            )
            # pass through video meta info
            results_per_vid['video_id'] = vidx
            results_per_vid['fps'] = fps
            results_per_vid['duration'] = vlen
            results_per_vid['feat_stride'] = stride
            results_per_vid['feat_num_frames'] = nframes
            results.append(results_per_vid)

        # step 3: postprocssing
        results = self.postprocessing(results)

        return results

    @torch.no_grad()
    def inference_single_video(
            self,
            points,
            fpn_masks,
            out_cls_logits,
            out_offsets,
            lb_logits_per_vid, rb_logits_per_vid
    ):
        """
            points: [(T, 4), (N1, 4), (N2, 4), (N3, 4), ...  (N(arch[2]), 4)]
            fpn_masks: [(T), (N1), (N2), (N3), ...  (N(arch[2]))]
            out_cls_logits: [(T, num_classes), (N1, num_classes), ...  (N(arch[2]), num_classes)] 
            out_offsets: [(T, 2 * (num_bins + 1)), (N1, 2 * (num_bins + 1)) ...  (N(arch[2]), 2 * (num_bins + 1))]
            out_lb_logits: [(num_classes, T), (num_classes, N1), ...  (num_classes, N(arch[2]))]
            out_rb_logits: [(num_classes, T), (num_classes, N1), ...  (num_classes, N(arch[2]))]
        """
        segs_all = []
        scores_all = []
        cls_idxs_all = []

        # loop over fpn levels
        for cls_i, offsets_i, pts_i, mask_i, sb_cls_i, eb_cls_i in zip(
                out_cls_logits, out_offsets, points, fpn_masks, lb_logits_per_vid, rb_logits_per_vid
        ):
            pred_prob = (cls_i.sigmoid() * mask_i.unsqueeze(-1)).flatten()

            # Apply filtering to make NMS faster following detectron2
            # 1. Keep seg with confidence score > a threshold
            keep_idxs1 = (pred_prob > self.test_pre_nms_thresh)
            pred_prob = pred_prob[keep_idxs1]
            topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

            # 2. Keep top k top scoring boxes only
            num_topk = min(self.test_pre_nms_topk, topk_idxs.size(0))
            pred_prob, idxs = pred_prob.sort(descending=True)
            pred_prob = pred_prob[:num_topk].clone()
            topk_idxs = topk_idxs[idxs[:num_topk]].clone()

            # fix a warning in pytorch 1.9
            pt_idxs = torch.div(
                topk_idxs, self.num_classes, rounding_mode='floor'
            )
            cls_idxs = torch.fmod(topk_idxs, self.num_classes)

            # 3. For efficiency, pad the boarder head with num_bins zeros (Pad left for start branch and Pad right
            # for end branch). Then we re-arrange the output of boundary branch to [class_num, T, num_bins + 1 (the
            # neighbour bin for each instant)]. In this way, the output can be directly added to the center offset
            # later.
            if self.use_trident_head:
                # pad the boarder
                x = (F.pad(sb_cls_i, (self.num_bins, 0), mode='constant', value=0)).unsqueeze(-1)  # pad left
                x_size = list(x.size())  # cls_num, T+num_bins, 1
                x_size[-1] = self.num_bins + 1
                x_size[-2] = x_size[-2] - self.num_bins  # cls_num, T, num_bins + 1
                x_stride = list(x.stride())
                x_stride[-2] = x_stride[-1]

                pred_start_neighbours = x.as_strided(size=x_size, stride=x_stride)

                x = (F.pad(eb_cls_i, (0, self.num_bins), mode='constant', value=0)).unsqueeze(-1)  # pad right
                pred_end_neighbours = x.as_strided(size=x_size, stride=x_stride)
            else:
                pred_start_neighbours = None
                pred_end_neighbours = None

            decoded_offsets = self.decode_offset(offsets_i, pred_start_neighbours, pred_end_neighbours)

            # pick topk output from the prediction
            if self.use_trident_head:
                offsets = decoded_offsets[cls_idxs, pt_idxs]
            else:
                offsets = decoded_offsets[pt_idxs]

            pts = pts_i[pt_idxs]

            # 4. compute predicted segments (denorm by stride for output offsets)
            seg_left = pts[:, 0] - offsets[:, 0] * pts[:, 3]
            seg_right = pts[:, 0] + offsets[:, 1] * pts[:, 3]

            pred_segs = torch.stack((seg_left, seg_right), -1)

            # 5. Keep seg with duration > a threshold (relative to feature grids)
            seg_areas = seg_right - seg_left
            keep_idxs2 = seg_areas > self.test_duration_thresh

            # *_all : N (filtered # of segments) x 2 / 1
            segs_all.append(pred_segs[keep_idxs2])
            scores_all.append(pred_prob[keep_idxs2])
            cls_idxs_all.append(cls_idxs[keep_idxs2])

        # cat along the FPN levels (F N_i, C)
        segs_all, scores_all, cls_idxs_all = [
            torch.cat(x) for x in [segs_all, scores_all, cls_idxs_all]
        ]
        results = {'segments': segs_all,
                   'scores': scores_all,
                   'labels': cls_idxs_all}

        return results

    @torch.no_grad()
    def postprocessing(self, results):
        # input : list of dictionary items
        # (1) push to CPU; (2) NMS; (3) convert to actual time stamps
        processed_results = []
        for results_per_vid in results:
            # unpack the meta info
            vidx = results_per_vid['video_id']
            fps = results_per_vid['fps']
            vlen = results_per_vid['duration']
            stride = results_per_vid['feat_stride']
            nframes = results_per_vid['feat_num_frames']
            # 1: unpack the results and move to CPU
            segs = results_per_vid['segments'].detach().cpu()
            scores = results_per_vid['scores'].detach().cpu()
            labels = results_per_vid['labels'].detach().cpu()
            if self.test_nms_method != 'none':
                # 2: batched nms (only implemented on CPU)
                segs, scores, labels = batched_nms(
                    segs, scores, labels,
                    self.test_iou_threshold,
                    self.test_min_score,
                    self.test_max_seg_num,
                    use_soft_nms=(self.test_nms_method == 'soft'),
                    multiclass=self.test_multiclass_nms,
                    sigma=self.test_nms_sigma,
                    voting_thresh=self.test_voting_thresh
                )
            # 3: convert from feature grids to seconds
            if segs.shape[0] > 0:
                segs = (segs * stride + 0.5 * nframes) / fps
                # truncate all boundaries within [0, duration]
                segs[segs <= 0.0] *= 0.0
                segs[segs >= vlen] = segs[segs >= vlen] * 0.0 + vlen
            # 4: repack the results
            processed_results.append(
                {'video_id': vidx,
                 'segments': segs,
                 'scores': scores,
                 'labels': labels}
            )

        return processed_results
