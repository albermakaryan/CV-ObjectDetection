import matplotlib.pyplot as plt
import numpy as np
import PIL
from helper import annotation_extractor
from icecream import ic
import os



ANNOTATION_PATH = "../../DATA/Data_COCO/annotations.coco.json"
IMAGE_PATH = "../../DATA/Data_COCO/images"

def show_image(annotation_file_path,number_of_images=12,*image_file_names):
        
        """
        Show the image.
        
        Parameters:
        -----------
        image: PIL.Image
            The image to be shown.
        
        """
        
        
        if len(image_file_names) == 0:
            
            # image_file_names = os.listdir(IMAGE_PATH)
            
            image_file_names = np.random.choice(os.listdir(IMAGE_PATH),size=number_of_images,replace=False)
            
        plt.figure(figsize=(20,20))
        

        for i in range(len(image_file_names)):
            
            image_file_name = image_file_names[i]
            
            image_path = os.path.join(IMAGE_PATH,image_file_name)
            
            annotations = annotation_extractor(annotation_file_path,image_file_name)
            ic(annotations)
            image = plt.imread(image_path)
            
            columns = 4 if number_of_images > 4 else number_of_images
            rows = number_of_images//columns if number_of_images%columns == 0 else number_of_images//columns + 1
            
            
            plt.subplot(rows,columns,i+1)
            plt.imshow(image)
            plt.grid(True)
            
            for i in range(len(annotations['label'])):
                
                bbox = annotations['bbox'][i]
                category = annotations['category'][i]
                label = annotations['label'][i]

                
                x_start = bbox[0]
                y_start = bbox[1]
                x_end = x_start + bbox[2]
                y_end = y_start + bbox[3]
                
                plt.plot([x_start,x_end,x_end,x_start,x_start],[y_start,y_start,y_end,y_end,y_start])
                plt.text(x_end,y_end,f"{label} : {category}")
                
        plt.show()     
        
        
        
        
        
if __name__ == "__main__":
    show_image(annotation_file_path=ANNOTATION_PATH,number_of_images=1)