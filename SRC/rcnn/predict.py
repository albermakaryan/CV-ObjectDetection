import torch
from icecream import ic 


@torch.no_grad()
def predict(image,model_path=None,model=None):
    
    model = torch.load(model_path) if model is None else model
    
    model.eval()
    prediction = model(image)
    
    return prediction
    
    
    
    
    
    
    pass

import numpy as np

def non_max_suppression(boxes, scores, threshold):
    # Sort boxes by scores in descending order
    order = np.argsort(scores)[::-1]

    keep = []
    while order.size > 0:
        # Pick the box with the highest confidence
        i = order[0]
        keep.append(i)

        # Calculate IoU with the remaining boxes
        overlaps = calculate_iou(boxes[i], boxes[order[1:]])

        # Discard boxes with high IoU
        inds = np.where(overlaps <= threshold)[0]
        order = order[inds + 1]

    return keep

def calculate_iou(box, boxes):
    # Calculate IoU between a box and an array of boxes
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - intersection

    iou = intersection / union
    return iou

# Example usage:
# boxes: Array of bounding boxes in format [x1, y1, x2, y2]
# scores: Confidence scores for each bounding box
# threshold: IoU threshold for NMS
# keep: Indices of the boxes to keep after NMS



@torch.no_grad()
def predict_all(device,model_path=None,model=None,batch=None,test_root="../DATA/Data/test",batch_size=2,threshold=0.3):
    
    
    model = torch.load(model_path) if model is None else model
    
    X, Y = batch
    X = [x.to(device) for x in X]

    with torch.no_grad():
        predictions = model(X)


    finall_predictsions = []
    for i in range(len(X)):
        
        boxes, scores, labels = [predictions[i][key].cpu().numpy() for key in ['boxes', 'scores', 'labels']]
              
        indecies = np.where(scores>threshold)
        
        if len(indecies[0]) == 0:
            continue
        boxes = boxes[indecies]
        scores = scores[indecies]
        labels = labels[indecies]
        
 
        finall_predictsions.append((boxes,scores,labels))
        
    return finall_predictsions
        
