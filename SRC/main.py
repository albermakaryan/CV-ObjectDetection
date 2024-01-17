
if __name__ == "__main__":
    
    from rcnn.model_training import train_model
    
    
    train_model(batch_size=2,lerning_rate=0.001,epochs=100,mode_save_root='trained_models',trainable_backbone_layers=5)