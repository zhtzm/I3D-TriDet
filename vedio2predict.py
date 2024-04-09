import time

import numpy as np

from actionIdentifyUI import finalControlShow
from i3d_feature_extraction.video_extract_couple_feature import video_feature
from TriDet.predict import predict

def vedio2predict(video_dir, config_file, ckpt_file, topk=10, level=0.25, mapping_file="data\\thumos\\mapping.json"):
    results = video_feature(video_dir, batch_size=40)
    predicts = predict(config=config_file, ckpt=ckpt_file, topk=topk, features=results, mapping_file=mapping_file, level=level)

    return predicts


def default_predict2show():
    video_dir = "tmp\\video"
    config = ".\\TriDet\\configs\\thumos_i3d.yaml"
    ckpt = "default_ckpt\\epoch_039.pth.tar"

    t1 = time.time()
    results = vedio2predict(video_dir, config, ckpt)
    print('总耗时: ', time.time() - t1, 's')

    for result in results:
        finalControlShow(result['segments'].tolist(),
                         result['scores'].tolist(),
                         result['labels'],
                         result['file'])
