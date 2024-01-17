from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def faster_rccn(freeze=False,trainable_backbone_layers = 3,number_of_classes=5):

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
    
    # Get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, number_of_classes) 

    # freeze all layers

    # if not freeze:
    #     for p in rcnn.parameters():
    #         p.requires_grad = True

    
    return model