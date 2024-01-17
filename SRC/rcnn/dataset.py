from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, \
        RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
import os
import sys
from icecream import ic
import PIL
import torch
import matplotlib.pyplot as plt
import cv2

# sys.path.append('../helper_functions')


from helper_functions.helper import annotation_extractor






class RCNN_Dataset(Dataset):
    
    def __init__(self,image_directory,annotation_file_path,transform=None):
        
        """
        Initialize the RCNN_Dataset class object.
        
        Parameters:
        -----------
        image_directory: str
            The directory where the images are stored.
        annotation_file_path: str
            The path to the annotation file.
        transform: torchvision.transforms
            The transforms to be applied to the images.
            
        """
        
        self.image_directory = image_directory
        self.files = os.listdir(self.image_directory)
        self.annotation_file_path = annotation_file_path
        
        self.annotations = []


        # self.images_id_file_path = {}
        
        if transform is None:
                self.transform = Compose([ToTensor(),
                                      Resize((224,224)),
                                      RandomHorizontalFlip(0.4),
                                      RandomVerticalFlip(0.4)])
                
                # self.transform = Compose([ToTensor()])

        else:
            
            self.transform = transform
        
        
    def __getitem__(self, index):
        
        
        """
        This method is used to get the image at the given index.
        
        Parameters:
        -----------
        index: int
            The index of the image to be retrieved.
        
        """
        
        img = self.files[index]
        file_path = os.path.join(self.image_directory,img)
        

        
        image = plt.imread(file_path) 
        image = cv2.resize(image,(224,224))/255
     
        image.shape = (3,224,224)
        # image = image.transpose((2, 0, 1))
        image = torch.as_tensor(image, dtype=torch.float32)
     
        # image = PIL.Image.open(file_path)
        # image = self.transform(image)
        
        annotations = annotation_extractor(image_file_name=img,
                                           annotation_file_path=self.annotation_file_path)
        
        
        old_height,old_width = annotations['height'],annotations['width']
        new_height,new_width = image.shape[1:]
        

        height_ratio = new_height/old_height
        width_ratio = new_width/old_width
        
        bbox = torch.tensor([[bbox[0]*width_ratio,bbox[1]*height_ratio,\
                            (bbox[0]+bbox[2])*width_ratio,(bbox[1]+bbox[3])*height_ratio] for bbox in annotations['bbox']])

       
        
        target = {}
        target['labels'] = torch.tensor(annotations['label'])
        
        target['boxes'] = bbox

        



        
        target['height'] = torch.tensor(new_height)
        target['width'] = torch.tensor(new_width)
        

        
    
           
        return image,target  


    
    def __len__(self):
        return len(self.files)
    
    
if __name__ == "__main__":
    
    ANNOTATION_PATH = "../../DATA/Data_COCO/annotations.coco.json"
    IMAGE_PATH = "../../DATA/Data_COCO/images"

    data = RCNN_Dataset(image_directory=IMAGE_PATH,annotation_file_path=ANNOTATION_PATH)
    
    n = len(data)
    
    for i in range(n):
        print(data[i])
        print("\n\n")
        break
        
    
    ic(n)
