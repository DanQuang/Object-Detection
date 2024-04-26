"""
Implementation of Yolo (v1) architecture
with slight modification with added BatchNorm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

"""
architecture config:
Tuple is structured by (kernel_size, filters, stride, padding)
"M" is simple max pooling with stride 2 and kernel_size 2x2
List is structured by tuples and lastly int with number of repeats 
"""

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 256, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    
]

class CNNBlocks(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlocks, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias= False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(num_features= out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class YOLOv1(nn.Module):
    def __init__(self, in_channels= 3, **kwargs):
        super(YOLOv1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcs = self.create_fcs(**kwargs)

    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim= 1))

    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlocks(
                    in_channels= in_channels, out_channels= x[1], kernel_size= x[0], stride = x[2], padding= x[3] 
                )]
                in_channels = x[1]
            elif type(x) == str:
                layers += [nn.MaxPool2d(kernel_size= 2, stride= 2)]

            elif type == list:
                conv1 = x[0]
                conv2 = x[1]
                
                for _ in range(x[2]):
                    layers += [
                        CNNBlocks(
                            in_channels,
                            conv1[1],
                            kernel_size = conv1[0],
                            stride = conv1[2],
                            padding = conv1[3]
                        ),
                        CNNBlocks(
                            conv1[1],
                            conv2[1],
                            kernel_size = conv2[0],
                            stride = conv2[2],
                            padding = conv2[3]
                        )
                    ]
                    in_channels = conv2[1]
    
        return nn.Sequential(*layers)

    def create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes

        # In original paper this should be
        # nn.Linear(1024*S*S, 4096),
        # nn.LeakyReLU(0.1),
        # nn.Linear(4096, S*S*(B*5+C))

        return nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(496),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.LazyLinear(S * S * (C + B * 5))
        )
    
def test(S= 7, B= 2, C= 20):
    model = YOLOv1(split_size= S, num_boxes= B, num_classes= C)
    x = torch.randn((2, 3, 448, 448))
    print(model(x).shape)