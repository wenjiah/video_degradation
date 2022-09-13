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
cd Detectors
python face_detection.py --base_name night-street
python face_detection.py --base_name UA-DETRAC
```

Detect *car* through Mask R-CNN for *night-street* video and store detection results in *./Data/filtered/night-street/freq002_res\[RESOLUTION\]_mrcnn/*. [Mask_RCNN](https://github.com/matterport/Mask_RCNN) is forked in the folder *./Detectors*. Install it as instructed in [Mask_RCNN](https://github.com/matterport/Mask_RCNN), and move *./Detectors/mask_rcnn.py* to *./Detectors/Mask_RCNN/mask_rcnn.py*.
```

```




