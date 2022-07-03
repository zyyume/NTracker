# NTracker: A New Object Tracker For Multiple Object Tracking

## Introduction
NTracker is a standalone tracker which can be asscioated with any object detector.
Below is an example of NTracker with [yolov5 detector](https://github.com/ultralytics/yolov5).

(Tested on windows 11, python 3.9.13, torch 1.11.0+cu113.)

<details open>
<summary>Installation</summary>

Install [cuda](https://developer.nvidia.com/cuda-downloads) and [pytorch](https://pytorch.org/)

```bash
### clone this repo
git clone https://github.com/zyyume/NTracker
cd NTracker
mkdir weights
mkdir output
  
### install yolov5
git clone https://github.com/ultralytics/yolov5  # clone
cd yolov5
pip install -r requirements.txt  # install
cd ..
```
</details>

<details open>
<summary>Directory Structure</summary>

download [yolov5 weights](https://github.com/ultralytics/yolov5/releases)

```bash
### install yolov5
D:\object_tracking
│   track.py # tracking script for yolov5
│   tracker.py # NTracker code
│
├───input
│       demo.jpg
│       demo.mp4
│
├───output # tracking output
├───weights # put downloaded weights here
│       yolov5x.pt
│       yolov5x6.pt
│
└───yolov5 # installed during installation step
```
</details>

<details open>
<summary>Run</summary>

```bash
### track a video
python .\track.py --weights .\weights\yolov5x6.pt --source .\input\demo1.mp4 --classes 0

### track webcam
python .\track.py --weights .\weights\yolov5x6.pt --source 0 --classes 0

### track a series of images
python .\track.py --weights .\weights\yolov5x6.pt --source .\input\img_folder --classes 0
```
</details>

<details open>
<summary>NTracker Configuration</summary>

modify track.py line 94

```python
f_threshold=12 # how many frames a disappeared track should exist
c_threshold=100 # confidence threshold for assigning tracks
### vp: weight of vector; wp: weight of width ; hp: weight iof height p; fp: weight of frame (reserved) 
vp=2; wp=0.25; hp=0.25; fp=0
```
</details>
