
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



def train(model,batch,optimizer,device):
    
    X,Y = batch
    X = [x.to(device) for x in X]
    Y = [{k:v.to(device) for k,v in y.items()} for y in Y]
    
    model.to(device)
    model.train()
    
    
    optimizer.zero_grad()
    losses = model(X,Y)


    loss = sum(loss for loss in losses.values())/len(X)
    

    loss.backward()
    optimizer.step()
    
    return loss    
    
    
    
def train_rccn(model,optimizer,train_dataloader,validation_dataloader,
               device,epochs=10,verbose=True):
    
    
    
    for i in range(1,epochs+1):

            
        iteration_train_loss = []
        
        for _, batch in enumerate(train_dataloader):
            
            train_loss = train(model=model,
                               batch=batch,
                               optimizer=optimizer,
                               device=device)
            iteration_train_loss.append(train_loss.item())
            
        iteration_validation_loss = []
        
        for _,batch in enumerate(validation_dataloader):
            
            validation_loss = validate(model=model,
                                       batch=batch,
                                       device=device)
            iteration_validation_loss.append(validation_loss.item())
            
        mean_train_loss = round(sum(iteration_train_loss)/len(iteration_train_loss),4)
        mean_validation_loss = round(sum(iteration_validation_loss)/len(iteration_validation_loss),4)
        
        if verbose:
        
            print(f"{i} / {epochs}, train_loss: {mean_train_loss}, validation_loss: {mean_validation_loss}\n")
            