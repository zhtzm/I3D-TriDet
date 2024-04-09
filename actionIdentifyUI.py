import cv2

letterWidth,letterLong=-20,-30
textSize=1
thickness=0


#x,y相对于右下角的坐标偏移(传入的值要为负)
def addTextToVideo(x, y, text, videoPathIn,segment):
    video_path = videoPathIn

    cap = cv2.VideoCapture(video_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter.fourcc(*'mp4v')

    startFrame=segment[0]*fps
    endFrame=segment[1]*fps
    frameCount=0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if startFrame <= frameCount <= endFrame:
            cv2.putText(frame, text[0], (width+x[0], height+y[0]), cv2.FONT_HERSHEY_COMPLEX, textSize, (0, 0, 255), thickness)
            cv2.putText(frame, text[1], (width+x[1], height+y[1]), cv2.FONT_HERSHEY_COMPLEX, textSize, (0, 0, 255), thickness)
            cv2.imshow('image', frame)
            cv2.waitKey(int(1000/fps))

        frameCount+=1

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

def secondControlShow(segment,score,label,file):
    text = [f"{segment[0]:.2f}"+"~"+f"{segment[1]:.2f}",
            label+":"+f"{score:.2f}"]
    x = [letterWidth*max(text[0].__len__(), text[1].__len__()), letterWidth*max(text[0].__len__(), text[1].__len__())]
    y = [letterLong*2, letterLong*1]
    addTextToVideo(x, y, text, file, segment)

def finalControlShow(segments,scores,labels,file):
    for i in range(len(labels)):
        secondControlShow(segments[i], scores[i], labels[i], file)


if __name__ == "__main__":
    segmentsSample=[[1.2,2.9],[7.1,9.1]]
    scoresSample=[0.3,0.5]
    labelsSmple=['walk','run']
    fileSample='videoSample.mp4'
    finalControlShow(segmentsSample,scoresSample,labelsSmple,fileSample)
