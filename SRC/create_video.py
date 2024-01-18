import torch
from rcnn.dataset import RCNN_Dataset
from torch.utils.data import DataLoader
from helper_functions.collate import collate_fn
from torchvision.io.video import write_video
import cv2
import json
import os
from icecream import ic
from rcnn.predict import predict_all

"""
Create a video from the predictions.

"""

with open("../DATA/Data_COCO/annotations.coco.json") as f:
    annotations = json.load(f)
    
    
classe_names = {category['id']:category['name'] for category in annotations['categories'] if category['id'] != 0}

    
TEST_ROOT = "../DATA/Data/test"
ANNOTATION_PATH = "../DATA/Data_COCO/annotations.coco.json"

batch_size = 1

test_dataset = RCNN_Dataset(image_directory=TEST_ROOT, annotation_file_path=ANNOTATION_PATH)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Load the model (ensure the model class is available in the code or imported properly)
model = torch.load('trained_models/rcnn_300_epoch_trained.pth')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()






    
frames = []

torch.cuda.empty_cache()

for ind, batch in enumerate(test_dataloader):
    
    X, Y = batch
    X = [x.to(device) for x in X]
    Y = [{k: v.to(device) for k, v in y.items()} for y in Y]


        
    predictions = predict_all(device,model=model,batch=batch,threshold=0.3)
    
    if len(predictions) == 0:
        continue

    for i in range(len(X)):
        
        
        prediction = predictions[i]
        
        
        
        old_height,old_width = int(Y[i]['old_height']),int(Y[i]['old_width'])
        new_height,new_width = int(Y[i]['new_height']),int(Y[i]['new_width'])
        
        
        size = 640
        width_ratio = 640/new_width
        height_ratio = 640/new_height
 
      
        image = X[i].cpu().numpy().reshape(new_width,new_height,3)*255
        image = cv2.resize(image, (size,size))

        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  

        height, width, _ = image.shape  

        

        image = image.astype(int)
        image = cv2.UMat(image)
        for box, score, label in zip(*prediction):
            

            x1, y1, x2, y2 = box
            x1,x2 = int(x1*width_ratio),int(x2*width_ratio)
            y1,y2 = int(y1*height_ratio),int(y2*height_ratio)
            
            
            start_point, end_point = (x1, y1), (x2, y2) 
            


            try:
                class_name = classe_names[label.item()]
            except KeyError:
                print(f"Error: Class label {label.item()} not found in classe_names dictionary.")
                continue
            


            image = cv2.rectangle(image, start_point, end_point, color=(0, 250, 0), thickness=2)
            image = cv2.putText(image, f'{class_name} : {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        frames.append(cv2.UMat.get((image)))
    # 


frames = torch.tensor(frames,dtype=torch.uint8)

output_file = 'output_video.mp4'

write_video(output_file, frames, fps=1,video_codec='h264')
