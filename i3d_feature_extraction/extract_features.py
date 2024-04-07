import os
import io
import zipfile
import torch
from torch.autograd import Variable
from PIL import Image
import numpy as np
from .pytorch_i3d import InceptionI3d
from torchvision import transforms


def load_zipframe(zipdata, name, transform):

    stream = zipdata.read(name)
    data = Image.open(io.BytesIO(stream))

    data = transform(data)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert(data.max()<=1.0)
    assert(data.min()>=-1.0)

    return data


def load_ziprgb_batch(rgb_zipdata, rgb_files, 
                   frame_indices, transform):

    batch_data = np.zeros(frame_indices.shape + (224,224,3))
    

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):

            batch_data[i,j,:,:,:] = load_zipframe(rgb_zipdata, 
                rgb_files[frame_indices[i][j]], transform)

    return batch_data


def load_zipflow_batch(flow_x_zipdata, flow_y_zipdata, 
                    flow_x_files, flow_y_files, 
                    frame_indices, transform):

    batch_data = np.zeros(frame_indices.shape + (224,224,2))

    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):

            batch_data[i,j,:,:,0] = load_zipframe(flow_x_zipdata, 
                flow_x_files[frame_indices[i][j]], transform)

            batch_data[i,j,:,:,1] = load_zipframe(flow_y_zipdata, 
                flow_y_files[frame_indices[i][j]], transform)

    return batch_data



def run(mode='rgb', load_model='', frequency=16, input_dir='', output_dir='', batch_size=40):
    transform = transforms.Compose([
        transforms.Resize(256),         
        transforms.CenterCrop(224),     
        ])
    chunk_size = 16

    assert(mode in ['rgb', 'flow'])
    
    if mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    
    i3d.load_state_dict(torch.load(load_model))
    i3d.replace_logits(400)
    i3d.cuda()
    i3d.eval() 

    def forward_batch(b_data):
        b_data = b_data.transpose([0, 4, 1, 2, 3])
        b_data = torch.from_numpy(b_data)   # b,c,t,h,w  # 40x3x16x224x224

        with torch.no_grad():
            b_data = Variable(b_data.cuda()).float()
            b_features = i3d.extract_features(b_data)
        
        b_features = b_features.data.cpu().numpy()[:,:,0,0,0]
        return b_features


    video_names = [i for i in os.listdir(input_dir) if i[0] == 'v']

    for video_name in video_names:

        save_file = '{}-{}.npy'.format(video_name, mode)
        if save_file in os.listdir(output_dir):
            continue

        frames_dir = os.path.join(input_dir, video_name)

        if mode == 'rgb':
            rgb_zipdata = zipfile.ZipFile(os.path.join(frames_dir, 'img.zip'), 'r')
            rgb_files = [i for i in rgb_zipdata.namelist() if i.startswith('img')]
            rgb_files.sort()
            frame_cnt = len(rgb_files)
        else:
            flow_x_zipdata = zipfile.ZipFile(os.path.join(frames_dir, 'flow_x.zip'), 'r')
            flow_x_files = [i for i in flow_x_zipdata.namelist() if i.startswith('x_')]

            flow_y_zipdata = zipfile.ZipFile(os.path.join(frames_dir, 'flow_y.zip'), 'r')
            flow_y_files = [i for i in flow_y_zipdata.namelist() if i.startswith('y_')]

            flow_x_files.sort()
            flow_y_files.sort()
            assert(len(flow_y_files) == len(flow_x_files))
            frame_cnt = len(flow_y_files)

        # clipped_length = (frame_cnt // chunk_size) * chunk_size   # Cut frames

        # Cut frames
        assert(frame_cnt > chunk_size)
        clipped_length = frame_cnt - chunk_size
        clipped_length = (clipped_length // frequency) * frequency  # The start of last chunk

        frame_indices = [] # Frames to chunks
        for i in range(clipped_length // frequency + 1):
            frame_indices.append(
                [j for j in range(i * frequency, i * frequency + chunk_size)])

        frame_indices = np.array(frame_indices)

        #frame_indices = np.reshape(frame_indices, (-1, 16)) # Frames to chunks
        chunk_num = frame_indices.shape[0]

        batch_num = int(np.ceil(chunk_num / batch_size))    # Chunks to batches
        frame_indices = np.array_split(frame_indices, batch_num, axis=0)

        full_features = [[]]

        for batch_id in range(batch_num):
            if mode == 'rgb':
                batch_data = load_ziprgb_batch(rgb_zipdata, rgb_files, 
                    frame_indices[batch_id], transform)
            else:
                batch_data = load_zipflow_batch(
                    flow_x_zipdata, flow_y_zipdata,
                    flow_x_files, flow_y_files, 
                    frame_indices[batch_id], transform)
                
            full_features[0].append(forward_batch(batch_data))

        full_features = [np.concatenate(i, axis=0) for i in full_features]
        full_features = [np.expand_dims(i, axis=0) for i in full_features]
        full_features = np.concatenate(full_features, axis=0)

        return full_features[0]


if __name__ == '__main__':
    run(mode='rgb', 
        load_model='./i3d_feature_extraction/models/rgb_imagenet.pt',
        sample_mode='resize',
        input_dir='./data/test/input_folder', 
        output_dir='./data/test/output_folder',
        batch_size=1,
        frequency=16)
