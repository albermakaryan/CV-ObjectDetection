import torch
from torch.utils.data import DataLoader

import sys
sys.path.append('./rcnn')
import os

from rcnn.train import train_rccn
from rcnn.model import faster_rccn
from rcnn.dataset import RCNN_Dataset
from helper_functions.collate import collate_fn
from icecream import ic

import warnings
warnings.filterwarnings('ignore')

TRAIN_ROOT = "../DATA/Data/train"
VALIDATION_ROOT = "../DATA/Data/validation"
TEST_ROOT = "../DATA/Data/test"
ANNOTAION_PATH = "../DATA/Data_COCO/annotations.coco.json"


batch_size = 1


train_set = RCNN_Dataset(image_directory=TRAIN_ROOT,annotation_file_path=ANNOTAION_PATH)
validation_set = RCNN_Dataset(image_directory=VALIDATION_ROOT,annotation_file_path=ANNOTAION_PATH)


train_dataloader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)
validation_dataloader = DataLoader(dataset=validation_set,batch_size=batch_size,shuffle=True,collate_fn=collate_fn)


# for i,batch in enumerate(train_dataloader):
#     X,Y = batch

#     ic(X)
#     ic(Y)
#     break
#     ic(Y[0]['labels'])
#     print()
# quit()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model = faster_rccn()
# model = get_object_detection_model()

optimizer = torch.optim.Adam(model.parameters(),lr=0.001)


epochs = 30

model_save_path = 'saved_models/rcnn_'+str(epochs)+ ' _epoch_trained.pth'

os.makedirs(model_save_path,exist_ok=True)

torch.cuda.empty_cache()

train_rccn(model=model,optimizer=optimizer,
           train_dataloader=train_dataloader,
           validation_dataloader=validation_dataloader,
           device=device,
           epochs=epochs)


torch.save(model,model_save_path)