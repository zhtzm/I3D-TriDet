import yaml

# 这是TriDet模型训练/测试使用到的所有设置的默认值
DEFAULTS = {
    # 事先设置好随机种子,确保实验在相同随机种子下得出相同的结论,这是为他人复现论文提供一个基础
    # 大的随机种子可以减少重复的概率
    "init_rand_seed": 1234567891,
    # 给出数据集的名称,因为代码中对不同的数据集的处理类不同
    "dataset_name": "epic",
    # GPU列表,这里是实现提供一个多GPU训练的设置,当然默认GPU数量为1
    "devices": ['cuda:0'],  
    # 这里给出的貌似是找训练特征文件的分割标签
    "train_split": ('training',),
    # 这里给出的貌似是找验证特征文件的分割标签
    "val_split": ('validation',),
    "model_name": "TriDet",
    "dataset": {
        # 隔多少帧取一个特征
        "feat_stride": 16,
        # 每个特征包含了多少帧的信息
        "num_frames": 32,
        # 这个是数据集视频原本的帧率信息,不同数据集可能不同,这里设置为None,表示去对应数据集josn文件中读取
        "default_fps": None,
        # 特征的纬度
        "input_dim": 2304,
        # 这里的数据集是分类数据集,这个实现表明有多少个类
        "num_classes": 97,
        # 特征的下采样率    如果设置为1,则使用原始分辨率;如果大于1,则对特征进行下采样以减少数据量
        "downsample_rate": 1,
        # 训练最长序列长度,Transformer里面就会统一特征长度
        "max_seq_len": 2304,
        # 如果一个动作的持续时间超过了trunc_thresh,则会认为该动作已经结束,并从该帧截断为一个新的动作片段
        "trunc_thresh": 0.5,
        # 特征随机裁剪的比例。设置为一个元组（例如(0.9, 1.0)）以启用随机特征裁剪，用于数据增强。这个参数可能不会被数据加载器实现。
        "crop_ratio": None,
        # 否强制将输入特征插值（upsampling）到固定大小
        # only used for ActivityNet
        "force_upsampling": False,
    },
    "loader": {
        "batch_size": 8,
        "num_workers": 4,
    },
    "model": {
        # type of backbone (SGP | conv)
        "backbone_type": 'SGP',
        # type of FPN (fpn | identity)
        "fpn_type": "identity",
        "backbone_arch": (2, 2, 5),
        # scale factor between pyramid levels
        # 特征金字塔每一层之间的scale变化的因子
        "scale_factor": 2,
        # 特征金字塔每一层回归视频段的长度
        "regression_range": [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)],
        # window size for self attention; <=1 to use full seq (ie global attention)
        "n_sgp_win_size": -1,
        # kernel size for embedding network
        "embd_kernel_size": 3,
        # (output) feature dim for embedding network
        "embd_dim": 512,
        # if attach group norm to embedding network
        "embd_with_ln": True,
        # feat dim for FPN
        "fpn_dim": 512,
        # if add ln at the end of fpn outputs
        "fpn_with_ln": True,
        # feat dim for head
        "head_dim": 512,
        # kernel size for reg/cls/center heads
        "head_kernel_size": 3,
        # kernel size for boundary heads
        "boudary_kernel_size": 3,
        # number of layers in the head (including the final one)
        "head_num_layers": 3,
        # if attach group norm to heads
        "head_with_ln": True,
        # defines the max length of the buffered points
        "max_buffer_len_factor": 6.0,
        # disable abs position encoding (added to input embedding)
        "use_abs_pe": False,
        # if use the Trident-head
        "use_trident_head": True,
        # how to downsample feature in FPN
        "downsample_type": "max",
        # the K in SGP
        "k": 1.5,
        # initialization of gaussian variance for the weight in SGP
        "init_conv_vars": 0,
        # the bin number in Trident-head (exclude 0)
        "num_bins": 16,
        # the power of iou weight in loss
        "iou_weight_power": 1.,
        # add gaussian noise with the variance, play a similar role to position embedding
        "input_noise": 0,

    },
    "train_cfg": {
        # radius | none (if to use center sampling)
        "center_sample": "radius",
        "center_sample_radius": 1.5,
        "loss_weight": 1.0,  # on reg_loss, use -1 to enable auto balancing
        "cls_prior_prob": 0.01,
        "init_loss_norm": 2000,
        # gradient cliping, not needed for pre-LN transformer
        "clip_grad_l2norm": -1,
        # cls head without data (a fix to epic-kitchens / thumos)
        "head_empty_cls": [],
        # dropout ratios for tranformers
        "dropout": 0.0,
        # ratio for drop path
        "droppath": 0.1,
        # if to use label smoothing (>0.0)
        "label_smoothing": 0.0,
    },
    "test_cfg": {
        "pre_nms_thresh": 0.001,
        "pre_nms_topk": 5000,
        "iou_threshold": 0.1,
        "min_score": 0.01,
        "max_seg_num": 1000,
        "nms_method": 'soft',  # soft | hard | none
        "nms_sigma": 0.5,
        "duration_thresh": 0.05,
        "multiclass_nms": True,
        "ext_score_file": None,
        "voting_thresh": 0.75,
    },
    # optimizer (for training)
    "opt": {
        # solver
        "type": "AdamW",  # SGD or AdamW
        # solver params
        "momentum": 0.9,
        "weight_decay": 0.0,
        "learning_rate": 1e-3,
        # excluding the warmup epochs
        "epochs": 30,
        # lr scheduler: cosine / multistep
        "warmup": True,
        "warmup_epochs": 5,
        "schedule_type": "cosine",
        # Minimum learning rate
        "eta_min": 1e-8,
        # in #epochs excluding warmup
        "schedule_steps": [],
        "schedule_gamma": 0.1,
    }
}


def _merge(src, dst):
    # 这个函数将默认值且不在config文件中添加上,但不把特殊设置的覆盖
    for k, v in src.items():
        if k in dst:
            # 如果是字典类型且在config文件里面,则递归检查
            if isinstance(v, dict):
                _merge(src[k], dst[k])
        else:
            # 如果默认值不在config文件中则添加
            dst[k] = v


def load_default_config():
    config = DEFAULTS
    return config


def _update_config(config):
    # fill in derived fields
    config["model"]["input_dim"] = config["dataset"]["input_dim"]
    config["model"]["num_classes"] = config["dataset"]["num_classes"]
    config["model"]["max_seq_len"] = config["dataset"]["max_seq_len"]
    config["model"]["train_cfg"] = config["train_cfg"]
    config["model"]["test_cfg"] = config["test_cfg"]
    return config


def load_config(config_file, defaults=DEFAULTS):
    with open(config_file, "r") as fd:
        # 这里面以yaml的格式导入config文件内容
        config = yaml.load(fd, Loader=yaml.FullLoader)
    # 补充默认值
    _merge(defaults, config)
    # 这个看函数是为model的字典里面添加一些特定的参数
    config = _update_config(config)
    return config
