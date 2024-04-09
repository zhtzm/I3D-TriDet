import os
import time
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F
from torch.autograd import Variable
from .pytorch_i3d import InceptionI3d
from torch.nn.parallel import DataParallel


RGB_MODEL = "i3d_feature_extraction\\models\\rgb_imagenet.pt"  
FLOW_MODEL = "i3d_feature_extraction\\models\\flow_imagenet.pt"  


def video_rgb_flow(video_file):
    assert os.path.exists(video_file)
    
    t1 = time.time()
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    img_frames = []  # 存储RGB帧的列表
    flow_x_frames = []  # 存储水平光流帧的列表
    flow_y_frames = []  # 存储垂直光流帧的列表

    # 初始化光流计算的变量
    prev_frame = None
    prev_gray = None

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        img_frames.append(frame)  # 将RGB帧添加到列表中

        # 计算光流
        if prev_frame is not None:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # 将光流转换为图像
            flow_x_img = np.clip(flow[..., 0] * 15 + 127.5, 0, 255).astype(np.uint8)
            flow_y_img = np.clip(flow[..., 1] * 15 + 127.5, 0, 255).astype(np.uint8)

            flow_x_frames.append(flow_x_img)  # 将水平光流帧添加到列表中
            flow_y_frames.append(flow_y_img)  # 将垂直光流帧添加到列表中

        prev_frame = frame
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()

    print(str(video_file) + ",rgb/flow提取完成,耗时: " + str(time.time() - t1) + "s")
    return fps, frame_count, duration, img_frames, flow_x_frames, flow_y_frames


def transform_data(data):
    resize = (256, 256)
    crop_size = 224
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = data.to(device)
    model = DataParallel(TransformModule(resize, crop_size)).to(device)
    
    transformed = model(data)
    transformed = transformed.type(torch.FloatTensor) 
    transformed = (transformed * 2 / 255) - 1
    assert(transformed.max() <= 1.0)
    assert(transformed.min() >= -1.0)
    return transformed

class TransformModule(torch.nn.Module):
    def __init__(self, resize, crop_size):
        super(TransformModule, self).__init__()
        self.resize = resize
        self.crop_size = crop_size
    
    def forward(self, data):
        transformed = []
        for frame in data:  
            batch_images = []
            for image in frame:
                resized_image = F.resize(image, self.resize)
                cropped_image = F.center_crop(resized_image, self.crop_size)
                batch_images.append(cropped_image)
            transformed.append(torch.stack(batch_images))
        transformed = torch.stack(transformed)
        return transformed


def video_feature(video_dir, frequency=4, batch_size=40, load_rgb_model=RGB_MODEL, load_flow_model=FLOW_MODEL):
    t0 = time.time()
    chunk_size = 16
    
    flow_i3d = InceptionI3d(400, in_channels=2)
    flow_i3d.load_state_dict(torch.load(load_flow_model))
    flow_i3d.replace_logits(400)
    flow_i3d.cuda()
    flow_i3d.eval() 

    rgb_i3d = InceptionI3d(400, in_channels=3)
    rgb_i3d.load_state_dict(torch.load(load_rgb_model))
    rgb_i3d.replace_logits(400)
    rgb_i3d.cuda()
    rgb_i3d.eval() 

    def forward_batch(i3d, b_data):
        with torch.no_grad():
            # print(b_data.shape)
            b_data = Variable(b_data.cuda())
            b_features = i3d.extract_features(b_data)
        
        b_features = b_features.data.cpu()[:,:,0,0,0]
        torch.cuda.empty_cache()
        return b_features

    video_names = [i for i in os.listdir(video_dir) if i.endswith('.mp4')]
    results = {}

    for video_name in video_names:
        video_file = os.path.join(video_dir, video_name)

        fps, frame_count, duration, img_frames, flow_x_frames, flow_y_frames = video_rgb_flow(video_file)
        t2 = time.time()
        flow_count = frame_count - 1
        
        assert(frame_count > chunk_size and flow_count > chunk_size)
        frame_clipped_length = frame_count - chunk_size
        flow__clipped_length = flow_count - chunk_size
        frame_clipped_length = (frame_clipped_length // frequency) * frequency  
        flow__clipped_length = (flow__clipped_length // frequency) * frequency  
        clipped_length = min(frame_clipped_length, flow__clipped_length)

        frame_indices = [] 
        for i in range(clipped_length // frequency + 1):
            frame_indices.append(
                [j for j in range(i * frequency, i * frequency + chunk_size)])
        frame_indices = np.array(frame_indices)
        chunk_num = frame_indices.shape[0]
        batch_num = int(np.ceil(chunk_num / batch_size))    
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)
        
        rgb_features = np.array(img_frames)
        flow_x_features = np.array(flow_x_frames)
        flow_y_features = np.array(flow_y_frames)

        rgb_full_features = []
        flow_full_features = []
        for batch_id in range(batch_num):
            rgb_batch_data = torch.tensor(rgb_features[frame_indices[batch_id]])
            flow_x_batch_data = flow_x_features[frame_indices[batch_id]]
            flow_y_batch_data = flow_y_features[frame_indices[batch_id]]
            combined_flow_data = torch.tensor(np.stack((flow_x_batch_data, flow_y_batch_data), axis=-1))
            rgb_batch_data = rgb_batch_data.permute(0, 1, 4, 2, 3)
            combined_flow_data = combined_flow_data.permute(0, 1, 4, 2, 3)

            new_rgb_batch_data = torch.zeros(rgb_batch_data.shape[:3] + (224, 224))
            new_combined_flow_data = torch.zeros(combined_flow_data.shape[:3] + (224, 224))

            new_rgb_batch_data = transform_data(rgb_batch_data) 
            new_combined_flow_data = transform_data(combined_flow_data)

            new_rgb_batch_data = new_rgb_batch_data.permute([0, 2, 1, 3, 4])
            new_combined_flow_data = new_combined_flow_data.permute([0, 2, 1, 3, 4])
                        
            rgb_full_features.append(forward_batch(rgb_i3d, new_rgb_batch_data))
            flow_full_features.append(forward_batch(flow_i3d, new_combined_flow_data))

        rgb_full_features = torch.cat(rgb_full_features, dim=0)
        flow_full_features = torch.cat(flow_full_features, dim=0)

        feature = torch.cat((rgb_full_features, flow_full_features), dim=1)

        results[video_name] = {"subset": "Predict",
                               "fps": fps, 
                               "duration": duration, 
                               "feature": feature,
                               "annotations": [],
                               "file": str(video_file)}
        print(str(video_name) + ",特征提取完成,耗时: " + str(time.time() - t2) + "s")

    del flow_i3d, rgb_i3d
    print(str(video_dir) + "特征提取完成,耗时: " + str(time.time() - t0) + "s")
    return results
    