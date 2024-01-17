import torch
from rcnn.dataset import RCNN_Dataset
from torch.utils.data import DataLoader
from helper_functions.collate import collate_fn
from torchvision.io.video import write_video
import cv2
import json
import os
from icecream import ic

# ... existing code ...

with open("../DATA/Data_COCO/annotations.coco.json") as f:
    annotations = json.load(f)
    
    
classe_names = {category['id']:category['name'] for category in annotations['categories'] if category['id'] != 0}

    
TEST_ROOT = "../DATA/Data/test"
ANNOTATION_PATH = "../DATA/Data_COCO/annotations.coco.json"

batch_size = 1

test_dataset = RCNN_Dataset(image_directory=TEST_ROOT, annotation_file_path=ANNOTATION_PATH)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# Load the model (ensure the model class is available in the code or imported properly)
model = torch.load('saved_models/rcnn_1_epoch_trained.pth')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()






    
frames = []

torch.cuda.empty_cache()

for ind, batch in enumerate(test_dataloader):
    
    X, Y = batch
    X = [x.to(device) for x in X]
    Y = [{k: v.to(device) for k, v in y.items()} for y in Y]

    # Perform inference and get predictions
    with torch.no_grad():
        predictions = model(X)
        
    # ic(predictions)
    # quit()
        
        

    # Iterate over each image in the batch
    for i in range(len(X)):
        
        
        image = test_dataset.files[i]
        image_path = os.path.join(TEST_ROOT,image)
        
        # print(image)       
        # print(i)
        # continue
        
      
        image = X[i].cpu().numpy().transpose((1, 2, 0))
        image = image*255
        image = image.astype('uint8')
        
        # ic(image)
        # quit()
        # Convert tensor to numpy array and make a copy
        
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        height, width, _ = image.shape  # Get image height and width

        # ic(predictions)
        # quit()
        # boxes, scores, labeles
        
        prediction = [predictions[i][key].cpu().numpy()[:5] for key in ['boxes', 'scores', 'labels']]

        # Draw bounding boxes on the image
        for box, score, label in zip(*prediction):
            
            
            # ic(box,score,label)
            # quit()
            
            x1, y1, x2, y2 = box.astype(int)  # Convert box coordinates to integers
            
            # start_point, end_point = (x1*width, y1*height), (x2*width, y2*height)
            
            
            
            start_point, end_point = (x1, y1), (x2, y2)

            try:
                class_name = classe_names[label.item()]
            except KeyError:
                print(f"Error: Class label {label.item()} not found in classe_names dictionary.")
                continue

            image = cv2.rectangle(image, start_point, end_point, color=(0, 255, 0), thickness=2)
            image = cv2.putText(image, f'{class_name} : {score:.2f}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        frames.append(image)
    # 
# quit()


# valid_frames = []
# for frame in frames:
    
#     if frame.shape[0] == 640 and frame.shape[1] == 640:
#         valid_frames.append(frame)


# valid_frames = torch.tensor(valid_frames,dtype=torch.uint8)
    
    
frames = torch.tensor(frames,dtype=torch.uint8)

# Define the output video file path
output_file = 'output_video.mp4'

# Write the frames to the video file
write_video(output_file, frames, fps=1,video_codec='h264')
