"""
Project structure:

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


"""