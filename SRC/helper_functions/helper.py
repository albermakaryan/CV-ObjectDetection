
import json
from icecream import ic
import torch



def annotation_extractor(image_file_name,
                         annotation_file_path,
                         model='rcnn'):

    """
    
    Read the annotation file and extract the annotations for the given image.
    
    
    
    
    Parameters:
    -----------
    annotation_file_path: str
        The path to the annotation file.
    image_file_name: str
        The name of the image file.
    model: str
        The model for which the annotations are to be extracted.
        
    
    
    
    return:
    -------
    image information: dict
    
    """

    # open the annotation file
    with open(annotation_file_path,'r') as f:
        
        annotations = json.load(f)
        
        
 


    if model == 'rcnn':
        
        # get the image id using image file name
        image = [image for image in annotations['images'] if image['file_name'] == image_file_name][0]
        
        
        image_id = image['id']
        
        # get the annotations for the given image id
        image_annotations = [annotation for annotation in annotations['annotations'] if annotation['image_id'] == image_id]
        
        
        output = dict()
        output['label'] = [annotation['category_id'] for annotation in image_annotations]
        output['bbox'] = [annotation['bbox'] for annotation in image_annotations]
        output['height'],output['width'] = image['height'],image['width']
        

        
        # image_annotation =  {annotation['category_id'] :annotation['bbox'] for annotation in annotations['annotations'] if annotation['image_id'] == image_id}

        
        output['category'] = [category['name'] for id in output['label']for category in annotations['categories'] if category['id'] == id ]
     
        # ic(annotations['categories'])
    
    
    return output

    
    
if __name__ == "__main__":
    
    ANNOTATION_PATH = "../../DATA/Data_COCO/annotations.coco.json"
    IMAGE_PATH = "../../DATA/Data_COCO/images"

    image = '100003_jpg.rf.f52e6e997c347d3ed2feb567faf4a5ef.jpg'
    
    ann = annotation_extractor(image_file_name=image,
                               annotation_file_path=ANNOTATION_PATH)
    
    
    print(ann)
    pass