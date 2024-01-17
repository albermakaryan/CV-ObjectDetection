import torch

@torch.no_grad()
def predict(image,model_path=None,model=None):
    
    model = torch.load(model_path) if model is None else model
    
    
    
    
    
    pass




def predict_all(model_path,test_root="../DATA/Data/test",batch_size=2):
    
    pass