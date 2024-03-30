# hydra models
from harp import vgg16_bn as HARP_Zhao2023Holistic_V16
from harp import ResNet18 as HARP_Zhao2023Holistic_R18

# harp models
from hydra import vgg16_bn as HYDRA_Sehwag2020Hydra_V16
from hydra import resnet18 as HYDRA_Sehwag2020Hydra_R18

from radmm import ResNet18 as resnet18_radmm
from radmm import VGG as vgg16_radmm

from twinrep import ResNet18 as resnet18_twinrep
from twinrep import vgg16 as vgg16_twinrep

from robustbird import ResNet18 as resnet18_robustbird
from robustbird import vgg16_bn as vgg16_robustbird

from flyingbird import ResNet18 as resnet18_flyingbird
from flyingbird import vgg16_bn as vgg16_flyingbird

# mad models
from mad import VGG as vgg16_mad
from mad import resnet as resnet18_mad

# li20 models
from li20 import ResNet18 as resnet18_li20
from li20 import vgg as vgg16_li20

# pwoa models
from pwoa import ResNet18 as resnet18_pwoa
from pwoa import VGG16 as vgg16_pwoa

from rst import vgg16_rst, resnet18_rst
from rst import vgg16_rst, resnet18_rst
# TODO: solve this RST thing

