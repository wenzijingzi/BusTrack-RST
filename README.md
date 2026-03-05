# BusTrack-RTS



## Highlights 🚀
- A training-free spatio-temporal enhancement framework for online MOT under mobile cameras
- Spatial–Temporal NMS stabilizes detections using short-term temporal support
- Foreground-aware GMC improves ego-motion estimation in bus-front scenarios
- Soft-constraint association enhances identity consistency under occlusion
- State-of-the-art performance on BusFrontMOT and competitive results on MOT17


## Abstract

Online multi-object tracking (MOT) under mobile camera conditions remains challenging due to unstable detections, motion prediction errors caused by camera ego-motion, and identity inconsistency under frequent occlusions. Most existing online trackers implicitly assume relatively stable camera viewpoints, which often leads to performance degradation when significant camera motion is present. To address this problem, we propose BusTrack-RST, a training-free and plug-and-play spatio-temporal enhancement framework for online MOT. The proposed framework improves tracking robustness by progressively suppressing uncertainty propagation at key stages of the tracking pipeline without modifying the detector architecture or requiring additional appearance training. Specifically, a Spatial–Temporal Non-Maximum Suppression (ST-NMS) module is introduced to stabilize detection sequences using short-term temporal consistency. A foreground-aware ego-motion compensation strategy is further designed to improve motion estimation by restricting feature matching to static background regions. In addition, a soft-constraint association strategy reshapes the matching cost through continuous regularization, enhancing identity consistency under occlusion and short-term missed detections. Experiments on the BusFrontMOT dataset and the MOT17 benchmark demonstrate that BusTrack-RST consistently improves MOTA, IDF1, and HOTA while maintaining near real-time performance.


### Visualization results on MOT challenge test set


https://user-images.githubusercontent.com/57259165/177045531-947d3146-4d07-4549-a095-3d2daa4692be.mp4

https://user-images.githubusercontent.com/57259165/177048139-05dcb382-010e-41a6-b607-bb2b76afc7db.mp4

https://user-images.githubusercontent.com/57259165/180818066-f67d1f78-515e-4ee2-810f-abfed5a0afcb.mp4

## Tracking performance
### Results on BusTrack-RTS test set
| Tracker       |  MOTA |  IDF1  |  HOTA  |
|:--------------|:-------:|:------:|:------:|
| BusTrack-RTS  |  77.29  | 81.76  | 72.67  |


### Results on MOT17 challenge test set
| Tracker       | MOTA   | IDF1 | HOTA |
|:--------------|:-------:|:------:|:------:|
|BusTrack-RTS   | 49.02   | 57.89 | 52.48 | 



## Installation

The code was tested on Windows11

BusTrack-RTS code is based on BoT-SORT. <br>
Visit their installation guides for more setup options.
 
### Setup with Anaconda
**Step 1.** Create Conda environment and install pytorch.
```shell
conda create -n bustrack-rts_env python=3.8
conda activate bustrack-rts_env
```
**Step 2.** Install torch and matched torchvision from [pytorch.org](https://pytorch.org/get-started/locally/).<br>
The code was tested using torch 1.11.0+cu113 and torchvision==0.12.0 

**Step 3.** Install BusTrack-RTS.
```shell
git clone https://github.com/wenzijingzi/BusTrack-RTS.git
cd BusTrack-RTS
pip3 install -r requirements.txt
python3 setup.py develop
```
**Step 4.** Install [pycocotools](https://github.com/cocodataset/cocoapi).
```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step 5. Others
```shell
# Cython-bbox
pip3 install cython_bbox

# faiss cpu / gpu
pip3 install faiss-cpu
pip3 install faiss-gpu
```

## Data Preparation

Download [BusTrack-RTS]([https://motchallenge.net/data/MOT17/](https://drive.google.com/drive/folders/1p8-WYcc3XkIidBscFnS022U1qBpyL-rS)). And put them in the following structure:

```
<dataets_dir>
      │
      ├── train
      │      ├── BF_01
      │      └── BF_02
      │      └── ```
      │      └── BF_19
``` 



 
## Model Zoo
Download and store the trained models in 'pretrained' folder as follow:
```
<BoT-SORT_dir>/pretrained
```
- We used the publicly available [ByteTrack](https://github.com/ifzhang/ByteTrack) model zoo trained on MOT17, MOT20 and ablation study for YOLOX object detection.

- Ours trained ReID models can be downloaded from [MOT17-SBS-S50](https://drive.google.com/file/d/1QZFWpoa80rqo7O-HXmlss8J8CnS7IUsN/view?usp=sharing), [MOT20-SBS-S50](https://drive.google.com/file/d/1KqPQyj6MFyftliBHEIER7m_OrGpcrJwi/view?usp=sharing).

- For multi-class MOT use [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) or [YOLOv7](https://github.com/WongKinYiu/yolov7) trained on COCO (or any custom weights). 


## Tracking

By submitting the txt files produced in this part to [MOTChallenge](https://motchallenge.net/) website and you can get the same results as in the paper.<br>
Tuning the tracking parameters carefully could lead to higher performance. In the paper we apply ByteTrack's calibration.

* **Detection**

```shell
cd <BusTrack-RTS_dir>
python3 tools/BFront_MOT/03temporal_nms_det_postprocess_X _all.py --default-parameters  --fuse
```

* **Tracking on BusTrack-RTS**

```shell
cd <BusTrack-RTS_dir>
python3 tools/BFront_MOT/02track_external_dets _all.py  --default-parameters --fuse
```

* **Evaluation**

```shell
cd <BusTrack-RTS_dir>

# BusTrack-RTS_dir
python3 tools/BFront_MOT/05eval_all_trackers_final_FPS.py  --default-parameters --fuse















