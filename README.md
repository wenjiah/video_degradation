# video_degradation
## Environment
All the experiments ran on Ubuntu 18.04.4 LTS on a 64-core (2.10GHz) Intel Xeon Gold 6130 server with 512 GB RAM and 4 GeForce GTX 1080 Ti GPUs.

These experiments require wget, Python 3.7, and Python libraries listed in requirements.txt.
```
pip3 install -r requirements.txt
```

## Step by Step Experiment Reproduction
### Step 1: Video Frame Preparation
Download and unzip *night-street* video with date stamp 2017-12-17 in the folder */z/wenjiah/video_degradation/data/svideo/night-street/*.
```
mkdir -p /z/wenjiah/video_degradation/data/svideo/night-street/
cd /z/wenjiah/video_degradation/data/svideo/night-street/
wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=1KhDzedVoiiWVD_pIJl4IGEeTi-QkO927' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KhDzedVoiiWVD_pIJl4IGEeTi-QkO927" -O 2017-12-17.zip && rm -rf /tmp/cookies.txt
unzip 2017-12-17.zip
```

Download and unzip *UA-DETRAC* video in the folder */z/wenjiah/video_degradation/data/svideo/UA-DETRAC/*.
```
mkdir -p /z/wenjiah/video_degradation/data/svideo/UA-DETRAC/
cd /z/wenjiah/video_degradation/data/svideo/UA-DETRAC/
wget http://detrac-db.rit.albany.edu/Data/DETRAC-test-data.zip
unzip DETRAC-test-data.zip
```

Select one out of every fifty frames from *night-street* video for our experiments and store them in */z/wenjiah/video_degradation/data/sample_frames/night-street/freq002_resori*, and select twelve sequences from *UA-DETRAC* video for our experiments and store them in */z/wenjiah/video_degradation/data/sample_frames/UA-DETRAC/freq1_resori/*.
```
mkdir -p /z/wenjiah/video_degradation/data/sample_frames/night-street/freq002_resori
mkdir -p /z/wenjiah/video_degradation/data/sample_frames/UA-DETRAC/freq1_resori/
cd ./Data
python gen_sample_frames.py --base_name night-street
python gen_sample_frames.py --base_name UA-DETRAC
```

Select two sequences MVI_40771 and MVI_40775 from *UA-DETRAC* video for profile sharing experiments.
```
mkdir -p /z/wenjiah/video_degradation/data/sample_frames/UA-DETRAC_share/
cd /z/wenjiah/video_degradation/data/svideo/UA-DETRAC/Insight-MVT_Annotation_Test
cp -r MVI_40771 /z/wenjiah/video_degradation/data/sample_frames/UA-DETRAC_share/
cp -r MVI_40775 /z/wenjiah/video_degradation/data/sample_frames/UA-DETRAC_share/
```

### Step 2: Frame Object Detection
**For convenience, we directly provide the detection results in *./Data/filtered*. These results can be obtained by running the following commands in this step.**

Detect *face* through MTCNN and store detection results in *./Data/filtered/night-street/freq002_face_mtcnn/* and *./Data/filtered/UA-DETRAC/freq1_face_mtcnn/*.
```
mkdir -p ./Data/filtered/night-street/freq002_face_mtcnn
mkdir -p ./Data/filtered/UA-DETRAC/freq1_face_mtcnn
cd ./Detectors
python face_detection.py --base_name night-street
python face_detection.py --base_name UA-DETRAC
```

[Mask_RCNN](https://github.com/matterport/Mask_RCNN) is forked in the folder *./Detectors*. Install it as instructed in [Mask_RCNN](https://github.com/matterport/Mask_RCNN), and move *./Detectors/mask_rcnn.py* to *./Detectors/Mask_RCNN/mask_rcnn.py*.

Detect *car* through Mask R-CNN for *night-street* video and store detection results in *./Data/filtered/night-street/freq002_res\[RESOLUTION\]_mrcnn/*.
```
mkdir -p ./Data/filtered/night-street/freq002_res640_mrcnn
mkdir -p ./Data/filtered/night-street/freq002_res576_mrcnn
mkdir -p ./Data/filtered/night-street/freq002_res512_mrcnn
mkdir -p ./Data/filtered/night-street/freq002_res448_mrcnn
mkdir -p ./Data/filtered/night-street/freq002_res384_mrcnn
mkdir -p ./Data/filtered/night-street/freq002_res320_mrcnn
mkdir -p ./Data/filtered/night-street/freq002_res256_mrcnn
mkdir -p ./Data/filtered/night-street/freq002_res192_mrcnn
cd ./Detectors/Mask_RCNN
python mask_rcnn.py --resolution 640
python mask_rcnn.py --resolution 576
python mask_rcnn.py --resolution 512
python mask_rcnn.py --resolution 448
python mask_rcnn.py --resolution 384
python mask_rcnn.py --resolution 320
python mask_rcnn.py --resolution 256
python mask_rcnn.py --resolution 192
```

[darknet](https://github.com/AlexeyAB/darknet) is forked in the folder *./Detectors*. Install it as instructed in [darknet](https://github.com/AlexeyAB/darknet), and move *./Detectors/darknet.py* to *./Detectors/darknet/darknet.py*.

Detect *car* and *person* through YOLOv4 for *night-street* and *UA-DETRAC* video and store detection results in *./Data/filtered/night-street/freq002_res\[RESOLUTION\]_yolo/*, *./Data/filtered/UA-DETRAC/freq1_res\[RESOLUTION\]_yolo/*, and *./Data/filtered/UA-DETRAC_share/\[SEQUENCEID\]_res\[RESOLUTION\]_yolo/*.
```
mkdir -p ./Data/filtered/night-street/freq002_res608_yolo
mkdir -p ./Data/filtered/night-street/freq002_res416_yolo
mkdir -p ./Data/filtered/night-street/freq002_res384_yolo
mkdir -p ./Data/filtered/night-street/freq002_res320_yolo
mkdir -p ./Data/filtered/night-street/freq002_res288_yolo
mkdir -p ./Data/filtered/night-street/freq002_res256_yolo
mkdir -p ./Data/filtered/night-street/freq002_res224_yolo
mkdir -p ./Data/filtered/night-street/freq002_res192_yolo
mkdir -p ./Data/filtered/night-street/freq002_res160_yolo
mkdir -p ./Data/filtered/night-street/freq002_res128_yolo
mkdir -p ./Data/filtered/UA-DETRAC/freq1_res608_yolo
mkdir -p ./Data/filtered/UA-DETRAC/freq1_res416_yolo
mkdir -p ./Data/filtered/UA-DETRAC/freq1_res384_yolo
mkdir -p ./Data/filtered/UA-DETRAC/freq1_res320_yolo
mkdir -p ./Data/filtered/UA-DETRAC/freq1_res288_yolo
mkdir -p ./Data/filtered/UA-DETRAC/freq1_res256_yolo
mkdir -p ./Data/filtered/UA-DETRAC/freq1_res224_yolo
mkdir -p ./Data/filtered/UA-DETRAC/freq1_res192_yolo
mkdir -p ./Data/filtered/UA-DETRAC/freq1_res160_yolo
mkdir -p ./Data/filtered/UA-DETRAC/freq1_res128_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40771_res608_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40771_res416_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40771_res384_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40771_res320_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40771_res288_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40771_res256_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40771_res224_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40771_res192_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40771_res160_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40771_res128_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40775_res608_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40775_res416_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40775_res384_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40775_res320_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40775_res288_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40775_res256_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40775_res224_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40775_res192_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40775_res160_yolo
mkdir -p ./Data/filtered/UA-DETRAC_share/40775_res128_yolo
```

Change resolution (width and height) in *./Detectors/darknet/cfg/yolov4.cfg* to 608.
```
cd ./Detectors/darknet
python darknet.py --resolution 608 --base_name night-street
python darknet.py --resolution 608 --base_name UA-DETRAC
python darknet.py --resolution 608 --base_name UA-DETRAC_share --shared_video_id 40771
python darknet.py --resolution 608 --base_name UA-DETRAC_share --shared_video_id 40775
```
Repeat this for other resolutions 416, 384, 320, 288, 256, 224, 192, 160, 128.

### Step 3: Run Experiments and Reproduce Plots
Run all the python files in *./Result*. They can produce plots stored in *./Result* with name consistent with figure numbers in the paper.
```
python Figure4a.py
python Figure4b.py
python Figure5.py
python Figure6a.py
python Figure6b.py
python Figure7.py
python Figure8.py
python Figure9.py
python Figure10.py
```

The true relative error and error bound can also be computed for each setting. Take the following case (aggregate function = AVG, video = night-street, reduced resolution = 384\*384, frame sampling fraction = 0.1, restricted class = face, correction set fraction = 0.5) as an example:
```
cd ./Estimation
python run_answer_quality_avg.py --obj_name car --base_name night-street --test_date 2017-12-17 --ground_truth freq002_res640_mrcnn --ground_truth_freq 0.02 --predictions freq002_res384_mrcnn --sample_frac 0.1 --constraints freq002_face_mtcnn --cons_obj_name face --use_val True --val_frac 0.5 --our_alg True
```
It takes only several seconds for this command, which contains 100 trials. It verifies that the estimation stage takes only tens of milliseconds for each set of degradation interventions as stated in Section 5.3.1.












