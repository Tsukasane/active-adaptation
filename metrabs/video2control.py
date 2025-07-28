import os
import cv2
import joblib
import numpy as np
import time

import tensorflow as tf
import tensorflow_hub as hub


import asyncio
from aiohttp import web
import cv2
import aiohttp
import numpy as np
import threading

import time
import torch
from collections import deque
from datetime import datetime
from torchvision import transforms as T
import time
from ultralytics import YOLO
import scipy.interpolate as interpolate

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

det_model = YOLO("yolov8s.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam

def download_model(model_type):
    server_prefix = 'https://omnomnom.vision.rwth-aachen.de/data/metrabs'
    model_zippath = tf.keras.utils.get_file(
        origin=f'{server_prefix}/{model_type}.zip',
        extract=True, cache_subdir='models')
    model_path = os.path.join(os.path.dirname(model_zippath), model_type)
    return model_path

def start_pose_estimate():
    global pose_mat, trans, dt, reset_offset, offset_height, superfast, j3d, j2d, num_ppl, bbox, frame, fps
    offset = np.zeros((5, 1))
    
    prev_box = None
    t_s = time.time()
    print('### Run Model...')
    
    # model = hub.load('https://bit.ly/metrabs_s') # or _l
    model = hub.load(r"./metrabs_dir")

    skeleton = 'smpl_24'
    joint_names = model.per_skeleton_joint_names[skeleton].numpy().astype(str)
    joint_edges = model.per_skeleton_joint_edges[skeleton].numpy()
    # viz = poseviz.PoseViz(joint_names, joint_edges)
    print("==================================> Metrabs model loaded <==================================")

    with torch.no_grad():
        while True:
            if not frame is None:
                pred = model.estimate_poses(
                            frame, 
                            tf.constant(bbox, dtype=tf.float32), 
                            skeleton=skeleton, 
                            default_fov_degrees=55, 
                            num_aug=1
                        )
                
                dt = time.time() - t_s
                fps = 1/dt
                # print(f"fps: {fps}")
                
                # camera = poseviz.Camera.from_fov(55, frame.shape[:2])
                # viz.update(frame, pred['boxes'], pred['poses3d'], camera)
                pred_j3d = pred['poses3d'].numpy()
                num_ppl = min(pred_j3d.shape[0], 5)
                
                j3d_curr = pred_j3d[:num_ppl]/1000
                if num_ppl < 5:
                    j3d[num_ppl:, 0, 0] = np.arange(5 - num_ppl) + 1
                    
                j2d =  pred['poses2d'].numpy()
                t_s = time.time()
                
                if reset_offset:
                    offset[:num_ppl] = - offset_height - j3d_curr[:num_ppl, [0], 1]
                    reset_offset = False
                
                j3d_curr[:offset.shape[0], ..., 1] += offset[:num_ppl]
                
                j3d = j3d.copy() # Trying to handle race condition
                j3d[:num_ppl] = j3d_curr

def commandline_input():
    global pose_mat, trans, dt, reset_offset, offset_height, superfast, j3d, j2d, num_ppl, bbox, frame, fps
    
    while True:
        command = input('Type a message to send to the server: ')
        if command == 'exit':
            print('Exiting!')
            raise SystemExit(0)
        elif command.startswith("r"):
            splits = command.split(":")
            if len(splits) > 1:
                offset_height = float(splits[-1])
            reset_offset = True
        elif command.startswith("fps"):
            print(fps)
        else:
            print("Unkonw command")

def frames_from_webcam():
    global frame, images_acc, recording, j2d, bbox
    cap = cv2.VideoCapture(0)
    prev_box = None
    
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame_orig = cap.read()
        if not ret:
            continue
        # x1, y1, x2, y2 = bbox
        detec_threshold = 0.6
        
        # h, w = frame_orig.shape[:2]
        # # only for zed2 camera
        # frame_orig = frame_orig[:, int(w/2):, :]

        frame = cv2.cvtColor(frame_orig, cv2.COLOR_BGR2RGB) # send to the detector & model 
        yolo_output = det_model.predict(source=frame, show=False, classes=[0], verbose=False)
        
        if len(yolo_output[0].boxes) > 0:
            yolo_out_xyxy = yolo_output[0].boxes.xyxy.cpu().numpy()
            
            bbox = np.stack([yolo_out_xyxy[:, 0], yolo_out_xyxy[:, 1], (yolo_out_xyxy[:, 2] - yolo_out_xyxy[:, 0]), (yolo_out_xyxy[:, 3] - yolo_out_xyxy[:, 1])], axis = 1)
            
            for i in range(len(yolo_out_xyxy)):
                x1, y1, x2, y2 = yolo_out_xyxy[i]
                frame_orig = cv2.rectangle(frame_orig, (int(x1), int(y1)), (int(x2), int(y2)), (154, 201, 219), 5)
            
        if not j2d is None:
            for pt in j2d.reshape(-1, 2):
                x, y = pt
                frame_orig = cv2.circle(frame_orig, (int(x), int(y)), 3, (255, 136, 132), 3)
                
        if recording:
            images_acc.append(frame_orig.copy())
            
        cv2.imshow('frame', frame_orig)
        
        if cv2.waitKey(1) == ord('q'):
            break
        # yield frame

async def pose_getter(request):
    # query env configurations
    global pose_mat, trans, dt, j3d, superfast
    curr_paths = {}
    if superfast:
        json_resp = {
            "j3d": j3d.tolist(),
            "dt": dt,
        }

    else:
        json_resp = {
            "pose_mat": pose_mat.tolist(),
            "trans": trans.tolist(),
            "dt": dt,
        }
        
    return web.json_response(json_resp)

def write_frames_to_video(frames, out_file_name = "output.mp4", frame_rate = 30, add_text = None, text_color = (255, 255, 255)):
    print(f"######################## Writing number of frames {len(frames)} ########################")
    if len(frames) == 0:
        return 
    y_shape, x_shape, _ = frames[0].shape
    out = cv2.VideoWriter(out_file_name, cv2.VideoWriter_fourcc(*'FMP4'), frame_rate, (x_shape, y_shape))
    transform_dtype = False
    transform_256 = False

    if frames[0].dtype != np.uint8:
        transform_dtype = True
    if np.max(frames[0]) < 1:
        transform_256 = True

    for i in range(len(frames)):
        curr_frame = frames[i]

        if transform_256:
            curr_frame = curr_frame * 256
        if transform_dtype:
            curr_frame = curr_frame.astype(np.uint8)
        if not add_text is None:
            cv2.putText(curr_frame, add_text , (0,  20), 3, 1, text_color)

        out.write(curr_frame)
    out.release()

reset_offset, offset_height = True, 0.92
images_acc, recording = deque(maxlen = 24000), False
bbox = np.zeros([5, 4])
pose_mat = np.zeros([24, 3, 3])
j3d = np.zeros([5, 24, 3])
j2d = None
trans = np.zeros([3])
dt = 1 / 10
fps = 0
num_ppl = 0

frame = None
superfast = True

app = web.Application(client_max_size=1024**2)
app.router.add_route('GET', '/get_pose', pose_getter)
threading.Thread(target=frames_from_webcam, daemon=True).start()
threading.Thread(target=start_pose_estimate, daemon=True).start()
threading.Thread(target=commandline_input, daemon=True).start()
print("=================================================================")
print("r: reset offset (use r:0.91), s: start recording, e: end recording, w: write video")
print("=================================================================")
web.run_app(app, port=8080)