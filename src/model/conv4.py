"""
Original file Copyright 2020 ElementAI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications made by Reece Walsh
"""


import torch
import numpy as np
import torch.nn.functional as F

class Conv4(torch.nn.Module):
    def __init__(self, num_classes=64, use_fc=False, prototype_size=1600):
        super().__init__()
        
        self.use_fc = use_fc

        self.adaptive_input = torch.nn.AdaptiveAvgPool2d(84)
        self.conv0 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(64)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(64)

        self.fc_prototype = torch.nn.Linear(1600, prototype_size)
        self.fc_classes = torch.nn.Linear(prototype_size, num_classes)

    def forward(self, x):
        x = self.adaptive_input(x)
        *dim, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.conv0(x) # 84
        x = F.relu(self.bn0(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 84 -> 42
        x = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 42 -> 21
        x = self.conv2(x)
        x = F.relu(self.bn2(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 21 -> 10
        x = self.conv3(x)
        x = F.relu(self.bn3(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 21 -> 5

        # if self.exp_dict["avgpool"] == True:
        #     x = x.mean(3, keepdim=True).mean(2, keepdim=True)

        x = torch.flatten(x, 1)
        x = self.fc_prototype(x)

        if self.use_fc:
            x = self.fc_classes(x)

        return x


class Conv4_Base(torch.nn.Module):
    def __init__(self, avgpool=True):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(64)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.avgpool = avgpool
        if avgpool:
            self.output_size = 64
        else:
            self.output_size = 1600
    
    def add_classifier(self, no, name="classifier", modalities=None):
        setattr(self, name, torch.nn.Linear(self.output_size, no))

    def forward(self, x, *args, **kwargs):
        *dim, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.conv0(x) # 84
        x = F.relu(self.bn0(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 84 -> 42
        x = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 42 -> 21
        x = self.conv2(x)
        x = F.relu(self.bn2(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 21 -> 10
        x = self.conv3(x)
        x = F.relu(self.bn3(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 21 -> 5
        if self.avgpool:
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)
        return x.view(*dim, self.output_size)


if __name__ == "__main__":
    model = Conv4(num_classes=64, use_fc=False, prototype_size=512)
    x = torch.randn(2, 3, 84, 84)
    y = model(x)
    print(y.shape)