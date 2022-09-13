from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import time
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base_name', required=True)
args = parser.parse_args()
base_name = args.base_name

if base_name == "night-street":
    detect_result = []
    sample_path = "/z/wenjiah/video_degradation/data/sample_frames/night-street/freq002_resori/"
    mtcnn = MTCNN(post_process=False, device='cuda:0', thresholds=[0.8, 0.8, 0.8]) # thresholds=[0.6, 0.7, 0.7]

    start = time.time()
    for index in np.arange(0,973101,50):
        img = Image.open(sample_path+"%d.jpg"%index)
        face = mtcnn(img)
        if(face is not None):
            detect_result.append([index, 'face'])

    print("total time:")
    print(time.time() - start)

    df = pd.DataFrame(detect_result, columns=['frame', 'object_name'])
    df.to_csv("../Data/filtered/night-street/freq002_face_mtcnn/night-street-2017-12-17.csv", index = None)

elif base_name == "UA-DETRAC":
    detect_result = []
    sample_path = "/z/wenjiah/video_degradation/data/sample_frames/UA-DETRAC/freq1_resori/"
    mtcnn = MTCNN(post_process=False, device='cuda:0', thresholds=[0.8, 0.8, 0.8])

    start = time.time()
    for index in np.arange(0,15210):
        img = Image.open(sample_path+"%d.jpg"%index)
        face = mtcnn(img)
        if(face is not None):
            detect_result.append([index, 'face'])

    print("total time:")
    print(time.time() - start)

    df = pd.DataFrame(detect_result, columns=['frame', 'object_name'])
    df.to_csv("../Data/filtered/UA-DETRAC/freq1_face_mtcnn/UA-DETRAC-2020-09-19.csv", index = None)

else:
    print("Please reenter video name.")