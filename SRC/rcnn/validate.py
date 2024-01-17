import torch


@torch.no_grad()
def validate(model,batch,device):
    
    X,Y = batch
    X = [x.to(device) for x in X]
    Y = [{k:v.to(device) for k,v in y.items()} for y in Y]
    
    model.to(device)
    model.train()
    
    
    losses = model(X,Y)
    

    loss = sum(loss for loss in losses.values())/len(X)
    
    return loss   