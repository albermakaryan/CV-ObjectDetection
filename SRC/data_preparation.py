
"""
Splits images from source root into three parts and saves each for given directory.
"""

if __name__ == "__main__":
    
    import os
    from helper_functions.split_roots import split
    
    working_directory = os.getcwd()

    DESTINATION_ROOT = "../DATA/Data"
    SOURCE_ROOT = "../DATA/Data_COCO/images"
    
    split(source_root=SOURCE_ROOT,train_size=0.8,
          validation_size=0.1,dest_root=DESTINATION_ROOT,random_seed=123,
          working_directory = working_directory)


