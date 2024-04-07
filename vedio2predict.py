import json
import os
import time

import numpy as np

from i3d_feature_extraction.conversion import extract_frames
from i3d_feature_extraction.extract_features import run
from TriDet.predict import predict


RGB_MODEL = "i3d_feature_extraction\\models\\rgb_imagenet.pt"  
FLOW_MODEL = "i3d_feature_extraction\\models\\flow_imagenet.pt"  


def get_feature(video_dir, input_dir, output_dir, load_rgb_model, load_flow_model):
    # "video_validation_0000051": {"subset": "Validation", "duration": 169.79, "fps": 30.0, "annotations": []}
    vedio_dict = {"database": {}}
    for file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, file)
        file_name, _ = os.path.splitext(file)
        tmp_path = os.path.join(input_dir, file_name)
        fps, duration = extract_frames(video_path, tmp_path)
        vedio_dict["database"][file_name] = {"subset": "Predict", "duration": duration, "fps": fps, "annotations": []}

        rgb_feature = run(mode='rgb', load_model=load_rgb_model, 
                        frequency=4, input_dir=input_dir, output_dir=output_dir)
        flow_feature = run(mode='flow', load_model=load_flow_model, 
                        frequency=4, input_dir=input_dir, output_dir=output_dir)
        feature = np.concatenate((rgb_feature, flow_feature), axis=1)
        np.save(os.path.join(output_dir, f'{file_name}.npy'), feature)

    file_path = os.path.join(output_dir, "feature.json")
    with open(file_path, "w") as json_file:
        json.dump(vedio_dict, json_file)


def clear_folder(folder_path):
    # 检查目标路径是否存在且是一个文件夹
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"Error: {folder_path} does not exist or is not a directory.")
        return

    # 遍历目标文件夹
    for root, dirs, files in os.walk(folder_path, topdown=False):
        # 删除文件
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f"Removed file: {file_path}")

        # 删除子文件夹
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)
            print(f"Removed directory: {dir_path}")


def vedio2predict(tmp_path, config_file, ckpt_file, topk=5, mapping_file="data\\thumos\\mapping.json", load_rgb_model=RGB_MODEL, load_flow_model=FLOW_MODEL):
    video_dir = os.path.join(tmp_path, "video")
    assert os.path.exists(video_dir)
    process_path = os.path.join(tmp_path, "tmp_process")
    predict_path = os.path.join(tmp_path, "tmp_predict")

    get_feature(video_dir=video_dir, input_dir=process_path, load_rgb_model=load_rgb_model, load_flow_model=load_flow_model, output_dir=predict_path)
    clear_folder(process_path)
    result = predict(config=config_file, ckpt=ckpt_file, topk=topk, predict_path=predict_path, video_path=video_dir, mapping_file=mapping_file)
    clear_folder(predict_path)

    return result


if __name__ == "__main__":
    tmp_path = "tmp"  
    config = "./TriDet/configs/thumos_i3d.yaml"
    ckpt = "./ckpt/thumos_i3d_pretrained/epoch_039.pth.tar"

    t1 = time.time()
    result = vedio2predict(tmp_path, config, ckpt)
    print(result)
    print(time.time() - t1)
