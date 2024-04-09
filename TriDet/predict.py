import os
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
import json

# our code
from TriDet.libs.core import load_config
from TriDet.libs.modeling import make_meta_arch
from TriDet.libs.utils import fix_random_seed


def process_input(data, downsample_rate, feat_stride, num_frames):
    processed = []
    for vd in data:
        feats = np.array(data[vd]["feature"])

        feats = feats[::downsample_rate, :]
        feat_stride = feat_stride * downsample_rate
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

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


def predict(config, ckpt, topk, features, mapping_file, level):
    if os.path.isfile(config):
        cfg = load_config(config)
    else:
        raise ValueError("Config file does not exist.")
    if ".pth.tar" in ckpt:
        assert os.path.isfile(ckpt), "CKPT file does not exist!"

    input_list = process_input(data=features, 
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
                    index = torch.where(tmp_output[vid_idx]['scores'] >= level)[0]
                    index = index[:topk] if len(index) < topk else index

                    tmp_output[vid_idx]['segments'] = tmp_output[vid_idx]['segments'][index]
                    tmp_output[vid_idx]['scores'] = tmp_output[vid_idx]['scores'][index]
                    tmp_output[vid_idx]['labels'] = tmp_output[vid_idx]['labels'][index]
                    english_labels = [mapping_data[str(label_id.item())] for label_id in tmp_output[vid_idx]['labels']]
                    tmp_output[vid_idx]['labels'] = english_labels
                    tmp_output[vid_idx]['file'] = os.path.join(features[tmp_output[vid_idx]['video_id']]["file"])
                    results.append(tmp_output[vid_idx])

    return results
