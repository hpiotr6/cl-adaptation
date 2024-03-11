from torchvision import models

from .lenet import LeNet
from .resnet32 import resnet32
from .resnet32_ln import resnet32_ln
from .resnet32_no_bn import resnet32_no_bn
from .resnet_custom import *
from .vggnet import VggNet

# available torchvision models
tvmodels = [
    "alexnet",
    "densenet121",
    "densenet169",
    "densenet201",
    "densenet161",
    "googlenet",
    "inception_v3",
    "mobilenet_v2",
    "mobilenet_v3_small",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "shufflenet_v2_x0_5",
    "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_5",
    "shufflenet_v2_x2_0",
    "squeezenet1_0",
    "squeezenet1_1",
    "vgg11",
    "vgg11_bn",
    "vgg13",
    "vgg13_bn",
    "vgg16",
    "vgg16_bn",
    "vgg19_bn",
    "vgg19",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "efficientnet_b0",
    "efficientnet_b1",
    "efficientnet_b2",
    "efficientnet_b3",
    "efficientnet_b4",
    "efficientnet_b5",
    "efficientnet_b6",
    "efficientnet_b7",
    "convnext_tiny",
]

allmodels = tvmodels + [
    "resnet32",
    "resnet32_no_bn",
    "resnet32_ln",
    "LeNet",
    "VggNet",
    *resnet_custom.__all__,
]


def set_tvmodel_head_var(model):
    match type(model):
        case (
            models.AlexNet
            | models.DenseNet
            | models.EfficientNet
            | models.MobileNetV2
            | models.ConvNeXt
        ):
            model.head_var = "classifier"
        case (
            models.Inception3
            | models.ResNet
            | models.VGG
            | models.SqueezeNet
            | models.MobileNetV3
            | models.GoogLeNet
            | models.ShuffleNetV2
        ):
            model.head_var = "fc"
        # case models.ConvNeXt:
        #     model.head_var = "head.fc"
        case _:
            raise ModuleNotFoundError
