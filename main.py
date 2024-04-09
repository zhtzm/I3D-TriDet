import time
from actionIdentifyUI import finalControlShow
from vedio2predict import vedio2predict


if __name__ == "__main__":
    video_dir = "tmp/video"
    config = "./TriDet/configs/thumos_i3d.yaml"
    ckpt = "epoch_039.pth.tar"

    t1 = time.time()
    results = vedio2predict(video_dir, config, ckpt)
    print('总耗时: ', time.time() - t1, 's')

    for result in results:
        finalControlShow(result['segments'].tolist(),
                         result['scores'].tolist(),
                         result['labels'],
                         result['file'])
        