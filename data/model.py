import torch.nn
import torch.nn as nn


def conv_link(in_dim, out_dim, s):
    conv = nn.Conv2d(in_dim, out_dim*4, (1, 1), stride=s)
    return conv

def identity_link():
    identity = nn.Identity()
    return identity


def conv_stage(in_dim, out_dim, s):
    model = nn.Sequential(
        nn.Conv2d(in_dim, in_dim, (1, 1), stride=s),
        nn.BatchNorm2d(in_dim),
        nn.ReLU(),
        nn.Conv2d(in_dim, in_dim, (3, 3), padding=1),
        nn.BatchNorm2d(in_dim),
        nn.ReLU(),
        nn.Conv2d(in_dim, out_dim * 4, (1, 1)),
        nn.BatchNorm2d(out_dim * 4)
    )
    return model


def identity_stage(in_dim):
    model = nn.Sequential(
        nn.Conv2d(in_dim, int(in_dim / 4), (1, 1)),
        nn.BatchNorm2d(int(in_dim / 4)),
        nn.ReLU(),
        nn.Conv2d(int(in_dim / 4), int(in_dim / 4), (3, 3), padding=1),
        nn.BatchNorm2d(int(in_dim / 4)),
        nn.ReLU(),
        nn.Conv2d(int(in_dim / 4), in_dim, (1, 1)),
        nn.BatchNorm2d(in_dim)
    )
    return model

def stage0():
    model = nn.Sequential(
        nn.Conv2d(3, 64, (7, 7), stride=(2, 2), padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d((3, 3), stride=2)
    )
    return model


class Resnet_50(nn.Module):
    def __init__(self):
        super(Resnet_50, self).__init__()
        self.identity_link = identity_link()
        self.stage0 = stage0()

        self.conv1_1 = conv_stage(64, 64, 1)
        self.conv_link1_1 = conv_link(64, 64, 1)
        self.identity1_2 = identity_stage(256)
        self.identity1_3 = identity_stage(256)

        self.conv2_1 = conv_stage(256, 128, 2)
        self.conv_link2_1 = conv_link(256, 128, 2)
        self.identity2_2 = identity_stage(512)
        self.identity2_3 = identity_stage(512)
        self.identity2_4 = identity_stage(512)

        self.conv3_1 = conv_stage(512, 256, 2)
        self.conv_link3_1 = conv_link(512, 256, 2)
        self.identity3_2 = identity_stage(1024)
        self.identity3_3 = identity_stage(1024)
        self.identity3_4 = identity_stage(1024)
        self.identity3_5 = identity_stage(1024)
        self.identity3_6 = identity_stage(1024)

        self.conv4_1 = conv_stage(1024, 512, 2)
        self.conv_link4_1 = conv_link(1024, 512, 2)
        self.identity4_2 = identity_stage(2048)
        self.identity4_3 = identity_stage(2048)
        self.relu = nn.ReLU()

        self.avgpool = nn.AvgPool2d((7, 7))
        self.flat = nn.Flatten()
        self.linear1 = nn.Linear(2048, 512)
        self.linear2 = nn.Linear(512, 64)
        self.linear3 = nn.Linear(64, 16)
        self.linear4 = nn.Linear(16, 2)

    def forward(self, x):
        output = self.stage0(x)

        output = self.conv1_1(output) + self.conv_link1_1(output)
        output = self.relu(output)
        output = self.identity1_2(output) + self.identity_link(output)
        output = self.relu(output)
        output = self.identity1_3(output) + self.identity_link(output)
        output = self.relu(output)

        output = self.conv2_1(output) + self.conv_link2_1(output)
        output = self.relu(output)
        output = self.identity2_2(output) + self.identity_link(output)
        output = self.relu(output)
        output = self.identity2_3(output) + self.identity_link(output)
        output = self.relu(output)
        output = self.identity2_4(output) + self.identity_link(output)
        output = self.relu(output)

        output = self.conv3_1(output) + self.conv_link3_1(output)
        output = self.relu(output)
        output = self.identity3_2(output) + self.identity_link(output)
        output = self.relu(output)
        output = self.identity3_3(output) + self.identity_link(output)
        output = self.relu(output)
        output = self.identity3_4(output) + self.identity_link(output)
        output = self.relu(output)
        output = self.identity3_5(output) + self.identity_link(output)
        output = self.relu(output)
        output = self.identity3_6(output) + self.identity_link(output)
        output = self.relu(output)

        output = self.conv4_1(output) + self.conv_link4_1(output)
        output = self.relu(output)
        output = self.identity4_2(output) + self.identity_link(output)
        output = self.relu(output)
        output = self.identity4_3(output) + self.identity_link(output)
        output = self.relu(output)

        output = self.avgpool(output)
        output = self.flat(output)
        output = self.linear1(output)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)

        return output

    def Init_weight(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0)
            elif isinstance(layer, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(layer.weight, 1)
                torch.nn.init.constant_(layer.weight, 0)
            elif isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, val=0.0)
