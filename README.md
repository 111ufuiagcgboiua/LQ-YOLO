# LQ-YOLO
This is an improved model based on YOLOv7, designed for object detection of two-wheeled vehicles on roads.  

## Code
The model code of LQ-YOLO is specifically presented in the '**models**' file.  

## Dataset
The dataset used in the research is decompressed into the '**two-wheelers**' file.  
Please note that the usage path of the training set is two-wheelers/train  
And check if the usage path is correct.  

## Platform
My platform is like this:
Windows 11  
NVIDIA 3070 gpu  
cuda 12.4  
python 3.9.19  
PyTorch 2.3.1  

## Model
The libraries that the model needs to be configured with can be downloaded from requirements.txt  
`pip install -r requirements.txt`

## Train
The command for model training   
`python train.py --workers 8 --device 0 --batch-size 8 --data data/data.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml`


