# LQ-YOLO  
- '**Enhanced Two-wheeler Detection via Lightweight Cross-scale Feature Fusion Network (LQ-YOLO)**'    
  by Yingjin Zhang, Longyu Ma et al, The Visual Computer
  
This is an improved model based on YOLOv7, designed for object detection of two-wheeled vehicles on roads. And mainly investigates the enhancement of the detection performance of two-wheeled vehicles through a lightweight detection framework and a multi-scale feature fusion module.  
    

## Code  
The data used in this paper is from the MS COCO2017 and the Objects21-vehicle datasets. You can choose to use the dataset employed in this study, and you can also use the official datasets for joint verification.   
Preparation of MS COCO2017 data.  
```
bash scripts/get_coco.sh
```  
The model code of LQ-YOLO is specifically presented in the '**models**' file.  
The "models" file contains the original files and the improved code. The improved parts can be found in files such as `common.py` and `yolo.py`.  

## Dataset  
The dataset used in the research is decompressed into the '**two-wheelers**' file.    
Please note that the usage path of the training set is `two-wheelers/train`  
And check if the usage path is correct.    

## Platform    
My platform is like this:    
- Windows 11    
- NVIDIA 3070 gpu    
- Cuda 12.4    
- python 3.9.19    
- PyTorch 2.3.1    

## Model    
The libraries that the model needs to be configured with can be downloaded from `requirements.txt `     
```
pip install -r requirements.txt
```

## Training  
The command for model training    

```
# train p5 models
python train.py --workers 8 --device 0 --batch-size 8 --data data/data.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
```
## Inference
The command for model detecting  

```
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
```

## Testing  
The command for model testing  
  
```
python test.py --data data/data.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7.pt --name yolov7_640_val
```  
