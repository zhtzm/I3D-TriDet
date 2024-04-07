# python imports
import argparse
import os
import time
import datetime
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)


################################################################################
def main(args):
    # 设置从第0个epoch开始训练
    args.start_epoch = 0
    # 下面的判断是为了加载config文件的正确性
    if os.path.isfile(args.config):
        # 加载配置文件并加上一些补全操作
        cfg = load_config(args.config)
    else:
        # 文件不存在或不是文件则报错
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # 判断设置中的输出父文件夹存不存在,不存在则创建
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    # 这里取出config文件的文件名,其实是用来命名输出config文件
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    # 下面这个判断主要是针对是否给出output文件夹名,初始化不同的模型保存文件夹名
    if len(args.output) == 0:
        ts = datetime.datetime.fromtimestamp(int(time.time()))
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    # 判断模型保存文件是否存在,不存在则创建
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    # 把所有有可能用到的库的随机种子都设置为给定的值,保证程序可重复性
    # rng_generator = torch.manual_seed(seed), 这里返回的仅是pytorch的随机生成器
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # 这里是针对多GPU对学习率和加载核心数进行调整,我们用不到
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    # 生成Dataset类
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # 下面的函数是自己创建的Dataset类里面的函数,用来获取特定的几个属性
    train_db_vars = train_dataset.get_attributes()
    # 添加参数
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # 生成DataLoader类
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    # 创建模型
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # 主要是将模型在多块GPU上动态训练
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # 创建优化器
    optimizer = make_optimizer(model, cfg['opt'])
    # 得到一个epoch有多少步,用来创建调度器
    num_iters_per_epoch = len(train_loader)
    # 创建调度器
    # 调度器用来动态调整学习率
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    # 这里是用来继续训练的,可以在以前训练的基础上继续训练,看起来是用来迁移训练用的,也可用于中断继续训练
    if args.resume:
        if os.path.isfile(args.resume):
            # load ckpt, reset epoch / best rmse
            checkpoint = torch.load(args.resume,
                                    map_location=lambda storage, loc: storage.cuda(cfg['devices'][0]))
            # 重新设置开始epoch
            args.start_epoch = checkpoint['epoch'] + 1
            # 加载ckpt的模型参数
            model.load_state_dict(checkpoint['state_dict'])
            # 给ema加载模型参数
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            # 加载优化器和调度器
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{:s}' (epoch {:d}".format(
                args.resume, checkpoint['epoch']
            ))
            # 在内存中删除,腾空间
            del checkpoint
        else:
            # 错误中断
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # 保存本次训练的config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    # 打印开始训练
    print("\nStart training model {:s} ...".format(cfg['model_name']))

    # 它首先尝试获取名为 'early_stop_epochs' 的键对应的值,如果不存在这个键,那么就会使用一个计算表达式
    max_epochs = cfg['opt'].get(
        'early_stop_epochs',
        cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
    )

    # 开始训练
    for epoch in range(args.start_epoch, max_epochs):
        # 训练一个epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema=model_ema,
            clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
            print_freq=args.print_freq
        )

        # 保存ckpt
        if (
                (epoch == max_epochs - 1)   # 最后一个epoch保存
                or
                (
                    (args.ckpt_freq > 0)and
                    (epoch > 0) and
                    (epoch % args.ckpt_freq == 0)   # 每ckpt_freq个epoch保存一次
                )
        ):
            save_states = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            save_states['state_dict_ema'] = model_ema.module.state_dict()

            # 保存ckpt
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch)
            )

    # 结束
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train a point-based transformer for action localization')
    # config文件地址
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)
