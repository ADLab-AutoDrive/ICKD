from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
import numpy as np

LAYER = {
         'resnet8': np.arange(1, (8 - 2) // 2 + 1),
         'resnet14': np.arange(1, (14 - 2) // 2 + 1),
         'resnet20': np.arange(1, (20 - 2) // 2 + 1),  # 9
         'resnet32': np.arange(1, (32 - 2) // 2 + 1),
         'resnet44': np.arange(1, (44 - 2) // 2 + 1),
         'resnet56': np.arange(1, (56 - 2) // 2 + 1),  # 27
         'resnet110': np.arange(2, (110 - 2) // 2 + 2, 1),  # 27
         'resnet8x4': np.arange(1, (8 - 2) // 2 + 1),
         'resnet32x4': np.arange(1, (32 - 2) // 2 + 1),
         'ResNet50': np.arange(1, 8),
         'wrn_16_1': np.arange(1, (16 - 4) // 2 + 1),  # 18
         'wrn_16_2': np.arange(1, (16 - 4) // 2 + 1),  # 6
         'wrn_40_1': np.arange(1, (40 - 4) // 2 + 1),  # 18
         'wrn_40_2': np.arange(1, (40 - 4) // 2 + 1),  # 6
         'resnet34': np.arange(1, (34 - 2) // 2 + 1),  # 16
         'resnet18': np.arange(1, (18 - 2) // 2 + 1),  # 8
         'resnet34im': np.arange(1, (34 - 2) // 2 + 1),  # 16
         'resnet18im': np.arange(1, (18 - 2) // 2 + 1),  # 8
         }

model_dict = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
}
