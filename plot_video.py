import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
from tqdm import tqdm
from loguru import logger


def concat_tile(im_list_2d):
    return cv2.vconcat([cv2.hconcat(im_list_h) for im_list_h in im_list_2d])

out_folder = "/Users/gouki/yolox_result_0224/plot_video"
os.makedirs(out_folder,exist_ok=True)
byte_folder = "/Users/gouki/yolox_result_0224/track_vis"
exp004_folder = "/Users/gouki/yolox_result_0224/exp004"
exp005_folder = "/Users/gouki/yolox_result_0224/exp005"
best_folder = "/Users/gouki/yolox_result_0224/best_ckpt"

byte_vids = sorted(glob(os.path.join(byte_folder,"*")))
exp004_vids = sorted(glob(os.path.join(exp004_folder,"*")))
exp005_vids = sorted(glob(os.path.join(exp005_folder,"*")))
best_vids = sorted(glob(os.path.join(best_folder,"*")))
for byte,exp004,exp005,best in zip(byte_vids,exp004_vids,exp005_vids,best_vids):
    byte_vid = cv2.VideoCapture(byte)
    exp004_vid = cv2.VideoCapture(exp004)
    exp005_vid = cv2.VideoCapture(exp005)
    best_vid = cv2.VideoCapture(best)
    width = best_vid.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = best_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = best_vid.get(cv2.CAP_PROP_FPS)
    video_name = os.path.basename(byte)
    logger.info("file: ".format(video_name))
    out_vid = cv2.VideoWriter(os.path.join(out_folder,video_name),cv2.VideoWriter_fourcc('M','J','P','G'),fps,(int(width*2), int(height*2)))
    while True:
        ret1, frame1 = byte_vid.read()
        ret2, frame2 = exp004_vid.read()
        ret3, frame3 = exp005_vid.read()
        ret4, frame4 = best_vid.read()
        if not ret1 or not ret2 or not ret3 or not ret4:
            break
        
        frame1 = cv2.putText(frame1,
            org=(int(width/2),25),
            text="ByteTrack",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_4) 
        frame2 = cv2.putText(frame2,
            org=(int(width/2),25),
            text="exp004",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_4)
        frame3 = cv2.putText(frame3,
            org=(int(width/2),25),
            text="exp005",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_4)
        frame4 = cv2.putText(frame4,
            org=(int(width/2),25),
            text="previous_model",
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_4)

        frame = concat_tile([[frame4,frame2],[frame3,frame1]])
        cv2.imshow('view', frame)

        out_vid.write(frame)

        k = cv2.waitKey(1)
        if k in [27, ord('q')]:
            break

    cv2.destroyAllWindows()
    out_vid.release()