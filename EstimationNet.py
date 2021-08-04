import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Estimation_direct(nn.Module):
    '''
    Noise estimator, with original 3 layers
    '''

    def __init__(self, input_channels=3, output_channels=3, num_of_layers=3, kernel_size=3, padding=1, features=64):
        super(Estimation_direct, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv3 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, 3, 1, 1)
        self.conv5 = nn.Conv2d(32, 3, 3, 1, 1)
        self.conv6 = nn.Conv2d(3, 3, 3, 1, 1)
        self.conv7 = nn.Conv2d(3, 3, 3, 1, 1)

        # self.avg_pool1 = nn.AvgPool2d(4, 4)
        self.avg_pool2 = nn.AvgPool2d(2, 2)

    def forward(self, input, alpha=0.8):
        x = input
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.avg_pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.avg_pool2(x)
        x = F.relu(self.conv5(x))
        x = F.upsample(x, [input.shape[2], input.shape[3]],
                       mode='bilinear')
        y = F.relu(self.conv6(x))
        y = F.relu(self.conv7(y))

        mixed = alpha * x + (1 - alpha) * y

        return mixed, x, y

