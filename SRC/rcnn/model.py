from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def faster_rccn(freeze=False,trainable_backbone_layers=3,number_of_classes=5):

    """
    Returns custom R-CNN model

    --------------------------

    freeze:
        whether freeze layers or not
    
    trainable_backbone_layers:
        how many layers' weigths set trainable from the end
    """

    model = fasterrcnn_resnet50_fpn_v2(weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
                                   trainable_backbone_layers=trainable_backbone_layers)
    

    
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feats,
                                                   number_of_classes)

    # freeze all layers
    if not freeze:
        for name, param in model.named_parameters():
            if trainable_backbone_layers > 0 and "backbone" in name:
                param.requires_grad = True
                trainable_backbone_layers -= 1
            else:
                param.requires_grad = False

    return model
    
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor 

# def get_object_detection_model(num_classes = 5, 
    #                            feature_extraction = True):
    # """
    # Inputs
    #     num_classes: int
    #         Number of classes to predict. Must include the 
    #         background which is class 0 by definition!
    #     feature_extraction: bool
    #         Flag indicating whether to freeze the pre-trained 
    #         weights. If set to True the pre-trained weights will be  
    #         frozen and not be updated during.
    # Returns
    #     model: FasterRCNN
    # """
    # # Load the pretrained faster r-cnn model.
    # model = fasterrcnn_resnet50_fpn(pretrained = True)    # If True, the pre-trained weights will be frozen.
    # if feature_extraction == True:
    #     for p in model.parameters():
    #         p.requires_grad = False    # Replace the original 91 class top layer with a new layer
    # # tailored for num_classes.
    # in_feats = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_feats,
    #                                                num_classes)    
    
    # return model