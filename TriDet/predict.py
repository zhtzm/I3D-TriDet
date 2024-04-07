import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import json

# our code
from TriDet.libs.core import load_config
from TriDet.libs.datasets import make_dataset, make_data_loader
from TriDet.libs.modeling import make_meta_arch
from TriDet.libs.utils import fix_random_seed


def process_input(data, predict_path, downsample_rate, feat_stride, num_frames):
    processed = []
    for vd in data:
        filename = os.path.join(predict_path, vd + '.npy')
        feats = np.load(filename).astype(np.float32)

        feats = feats[::downsample_rate, :]
        feat_stride = feat_stride * downsample_rate
        feat_offset = 0.5 * num_frames / feat_stride

        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # if vd['segments'] is not None:
        #     vd['segments'] = np.array([[(start_end * vd['fps'] / feat_stride - feat_offset) for start_end in segment] 
        #                                for segment in vd['segments']])
        #     segments = torch.from_numpy(vd['segments'])
        #     labels = torch.from_numpy(np.array(vd['labels']))
        # else:
        #     segments, labels = None, None

        segments, labels = None, None

        data_dict = {'video_id'        : vd,
                     'feats'           : feats,      
                     'segments'        : segments,   
                     'labels'          : labels,     
                     'fps'             : data[vd]['fps'],
                     'duration'        : data[vd]['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : num_frames}

        processed.append(data_dict)
    
    return processed


def predict(config, ckpt, topk, predict_path, video_path, mapping_file):
    if os.path.isfile(config):
        cfg = load_config(config)
    else:
        raise ValueError("Config file does not exist.")
    if ".pth.tar" in ckpt:
        assert os.path.isfile(ckpt), "CKPT file does not exist!"

    json_path = os.path.join(predict_path, "feature.json")
    assert os.path.exists(json_path)
    with open(json_path, 'r', encoding='utf-8') as f:
        json_path = json.load(f)
    input_data = json_path['database']
    input_list = process_input(data=input_data, 
                               predict_path=predict_path, 
                               downsample_rate=cfg["dataset"]['downsample_rate'],
                               feat_stride=cfg["dataset"]['feat_stride'],
                               num_frames=cfg["dataset"]['num_frames'])

    _ = fix_random_seed(0, include_cuda=True)
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    print("=> loading checkpoint '{}'".format(ckpt))
    checkpoint = torch.load(
        ckpt,
        map_location = lambda storage, loc: storage.cuda(cfg['devices'][0])
    )
    model.load_state_dict(checkpoint['state_dict_ema'])
    del checkpoint

    with open(mapping_file, "r") as map:
        mapping_data = json.load(map)

    model.eval()
    results = []
    for one_input in input_list:
        with torch.no_grad():
            tmp_output = model([one_input])
            num_vids = len(tmp_output)
            for vid_idx in range(num_vids):
                if tmp_output[vid_idx]['segments'].shape[0] > 0:
                    tmp_output[vid_idx]['segments'] = tmp_output[vid_idx]['segments'][:topk]
                    tmp_output[vid_idx]['scores'] = tmp_output[vid_idx]['scores'][:topk]
                    tmp_output[vid_idx]['labels'] = tmp_output[vid_idx]['labels'][:topk]
                    english_labels = [mapping_data[str(label_id.item())] for label_id in tmp_output[vid_idx]['labels']]
                    tmp_output[vid_idx]['labels'] = english_labels
                    video_name = f"{tmp_output[vid_idx]['video_id']}" + ".mp4"
                    tmp_output[vid_idx]['file'] = os.path.join(video_path, video_name)
                    results.append(tmp_output[vid_idx])

    return results
