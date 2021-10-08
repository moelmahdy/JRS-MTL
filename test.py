from torchsummary import summary

from graphs.losses.loss import *
from graphs.models.seddnet import SEDDNet
from graphs.models.segnet import SegNet

model = SegNet(in_channels=1, classes=5, depth=3,
               initial_channels=None, channels_list=[23, 45, 91]).to('cpu')

model = SEDDNet(in_channels=2, depth=3, initial_channels=None, channels_list=[23, 45, 91]).to('cuda')

summary(model, [(1, 96, 96, 96), (1, 96, 96, 96)])
input = torch.rand((1, 1, 96, 96, 96))# 1x1x56x56x56
res = model(input)


print(4)
