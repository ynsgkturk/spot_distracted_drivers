import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    """
        >>>conv_block = conv_block(3, 64)
        >>>noise = torch.rand((32, 3, 64, 64))
        >>>out = conv_block(noise)
        >>>out.size()

        >>>torch.Size([1, 64, 32, 32])
    """
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=2, padding="same"),
              nn.ReLU(),
              nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(2),
              ]
    return nn.Sequential(*layers)


class DistractedDriverDetectionModel(nn.Module):
    def __init__(self, input_size=3, num_classes=10):
        super(DistractedDriverDetectionModel, self).__init__()
        self.model = nn.Sequential(
            conv_block(in_channels=input_size, out_channels=64),
            conv_block(in_channels=64, out_channels=128),
            conv_block(in_channels=128, out_channels=256),
            conv_block(in_channels=256, out_channels=512),
        )

        self.fully_connected_layer = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 500),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(500, num_classes),
            nn.Softmax(1),
        )

    def forward(self, x):
        output = self.model(x)
        output = self.fully_connected_layer(output)
        return output
