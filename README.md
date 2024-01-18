# Computer Vision: Object Detection
## About the project

In this project, is used data from [Roboflow](https://universe.roboflow.com/spark-intelligence-scqhh/bottle-defect-detection) (COCO format).
to fine-tune a Faster R-CNN model from torchvision.

The aim of these project is to train model, which will predict bottle which are with no label, no cap and crumbled.
The model is trained 300 epochs with a batch size of 2 and a learning rate of 0.001.

### 1. Project structure

    Additional:
        CV_DL_Final_task.pdf
    DATA
        Data
            train
                000000000009.jpg
                000000000025.jpg
                ...
            validation
                000000000139.jpg
                000000000285.jpg
                ...
            test
                000000000139.jpg
                000000000285.jpg
                ...
        Data_COCO
            annotations.coco.json
            images
                000000000009.jpg
                000000000025.jpg
                ...
            README.dataset.txt
            README.roboflow.txt
            
        .data.yaml.swp
    SRC
    
        readme.py
        main.py
        data_preparation.py
        create_video.py
        rcnn
            __init__.py
            
            dataset.py
            model_training.py
            predict.py
            train.py
            validate.py
            model.py
            visualization.py
            
        helper_functions
            __init__.py
            
            collate.py
            split_roots.py
            visual.py
            helper.py
            

    README.md
    requirements.txt
    setup.py
    setup.cfg
    .gitignore
    .gitattributes




