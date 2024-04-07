import shutil
import cv2
import os
import numpy as np
import zipfile


# Integrated function to extract frames and optical flow
def extract_frames(video_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if (os.path.exists(os.path.join(output_dir, 'img.zip')) and
        os.path.exists(os.path.join(output_dir, 'flow_x.zip')) and
        os.path.exists(os.path.join(output_dir, 'flow_y.zip'))):
        print("Have created")
        return
    
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / fps

    # Create output directories
    img_dir = os.path.join(output_dir, 'img')
    flow_dir = os.path.join(output_dir, 'flow')

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(flow_dir, exist_ok=True)

    # Initialize variables for optical flow calculation
    prev_frame = None
    prev_gray = None

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(img_dir, f'img_{i:05d}.jpg')
        cv2.imwrite(frame_path, frame)

        # Calculate optical flow
        if prev_frame is not None:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

            # 将光流转换为图像
            flow_x_img = np.clip(flow[..., 0] * 15 + 127.5, 0, 255).astype(np.uint8)
            flow_y_img = np.clip(flow[..., 1] * 15 + 127.5, 0, 255).astype(np.uint8)

            flow_x_path = os.path.join(flow_dir, f'x_{i:05d}.jpg')
            flow_y_path = os.path.join(flow_dir, f'y_{i:05d}.jpg')

            cv2.imwrite(flow_x_path, flow_x_img)
            cv2.imwrite(flow_y_path, flow_y_img)

        prev_frame = frame
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cap.release()

    # Zip the frames and flow files
    with zipfile.ZipFile(os.path.join(output_dir, 'img.zip'), 'w') as img_zip:
        for root, dirs, files in os.walk(img_dir):
            for file in files:
                img_zip.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), img_dir))

    with zipfile.ZipFile(os.path.join(output_dir, 'flow_x.zip'), 'w') as flow_x_zip:
        for root, dirs, files in os.walk(flow_dir):
            for file in files:
                if file.startswith('x_'):
                    flow_x_zip.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), flow_dir))

    with zipfile.ZipFile(os.path.join(output_dir, 'flow_y.zip'), 'w') as flow_y_zip:
        for root, dirs, files in os.walk(flow_dir):
            for file in files:
                if file.startswith('y_'):
                    flow_y_zip.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), flow_dir))

    try:
        shutil.rmtree(img_dir)
        shutil.rmtree(flow_dir)
    except OSError as e:
        print(f"Error: {e.strerror}")

    print("Extraction completed.")
    return fps, duration_seconds