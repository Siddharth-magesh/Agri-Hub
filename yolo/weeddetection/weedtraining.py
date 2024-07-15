#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[4]:


import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Define the class names
class_names = [
    "Carpetweeds",
    "Crabgrass",
    "Eclipta",
    "Goosegrass",
    "Morningglory",
    "Nutsedge",
    "Palmeramaranth",
    "Pricklysida",
    "Purslane",
    "Ragweed",
    "Sicklepod",
    "Spottedspurge",
    "Spurredanoda",
    "Swinecress",
    "Waterhemp"
]

# Load the YOLO model
model = YOLO(r"P:\SmartHacks\todo\yolo\weeddetection\runs\detect\train\weights\last.pt")

# Perform inference

def weed_detection(image_path):
    # Load image
    img = cv2.imread(image_path)
    # Perform inference
    results = model(img)
    
    # Check the structure of the results
    if not results:
        print("No results found.")
        return
    
    # Extract bounding boxes, class ids, and confidence scores
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        # Draw bounding boxes and labels on the image
        for box, class_id, confidence in zip(boxes, class_ids, confidences):
            label = f"{class_names[class_id]} {confidence:.2f}"
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            

    # Convert BGR image to RGB for displaying with matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    weed_predicted=class_names[class_id]
    return(weed_predicted)






