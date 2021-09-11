# Yolo_Object_Detection

Inspiration of this project is taken from Deep Learning Specialiasation coursera and guided project on coursera. This Real Time object detection is apply on images as well as videos.
 

## Dependencies:
    Opencv, python
    
 ## Uses:
  Can use this project through command line:
  
  ### **Real Time object detection in video**  
  > python yolo.py -w=yolov3-tiny.weights -cfg=cfg/yolov3-tiny.cfg -v=Marketstreet.mp4 -l=data/coco.names -c=0.4 -t=0.4 .
   
   - -w: option for pre-trained weights
   - -cfg: configuration file
   - -v: video input
   - -c: confidence threshold
   - -t: threshold for non-max suppression
   - -vo: video output path
   
   shorthand version:
   > python yolo.py -v=images/Marketstreet.mp4.

  ### **Object detection in image**
  > python yolo.py -w=YOLOv3/yolov3-tiny.weights -cfg=YOLOv3/yolov3-tiny.cfg -i=images/fruit.jpg -l=YOLOv3/coco.names -c=0.4 -t=0.4    
  
  - -i: image input path
    
    Shorthand version:
   > python yolo.py -i=images/fruit.jpg
   

# References
Guided project ***Perform Real-Time Object Detection with YOLOv3*** from Coursera. 

  
