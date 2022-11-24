"""
Copyright 2020 ElementAI

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications by Reece Walsh
"""

import torch
import numpy as np
import torch.nn.functional as F

class Block(torch.nn.Module):
    def __init__(self, ni, no, stride, dropout=0, groups=1):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(dropout) if dropout > 0 else lambda x: x
        self.conv0 = torch.nn.Conv2d(ni, no, 3, stride, padding=1, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(no)
        self.conv1 = torch.nn.Conv2d(no, no, 3, 1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(no)
        self.conv2 = torch.nn.Conv2d(no, no, 3, 1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(no)
        if stride == 2 or ni != no:
            self.shortcut = torch.nn.Conv2d(ni, no, 1, stride=1, padding=0)

    def get_parameters(self):
        return self.parameters()

    def forward(self, x, is_support=True):
        y = F.relu(self.bn0(self.conv0(x)), True)
        y = self.dropout(y)
        y = F.relu(self.bn1(self.conv1(y)), True)
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x), True)


class Resnet12(torch.nn.Module):
    def __init__(self, width, dropout, num_classes=64, use_fc=False):
        super().__init__()
        self.output_size = 512
        assert(width == 1) # Comment for different variants of this model
        self.widths = [x * int(width) for x in [64, 128, 256]]
        self.widths.append(self.output_size * width)
        self.bn_out = torch.nn.BatchNorm1d(self.output_size)

        start_width = 3
        for i in range(len(self.widths)):
            setattr(self, "group_%d" %i, Block(start_width, self.widths[i], 1, dropout))
            start_width = self.widths[i]
        
        self.use_fc = use_fc
        self.add_classifier(num_classes)

    def add_classifier(self, nclasses, name="classifier", modalities=None):
        setattr(self, name, torch.nn.Linear(self.output_size, nclasses))

    def up_to_embedding(self, x):
        """ Applies the four residual groups
        Args:
            x: input images
            n: number of few-shot classes
            k: number of images per few-shot class
            is_support: whether the input is the support set (for non-transductive)
        """
        for i in range(len(self.widths)):
            x = getattr(self, "group_%d" % i)(x)
            x = F.max_pool2d(x, 3, 2, 1)
        return x

    def forward(self, x):
        """Main Pytorch forward function
        Returns: class logits
        Args:
            x: input mages
            is_support: whether the input is the sample set
        """
        *args, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.up_to_embedding(x)
        x = F.relu(self.bn_out(x.mean(3).mean(2)), True)
        if self.use_fc:
            return self.classifier(x)
        else:
            return x

if __name__ == "__main__":
    import torch
    model = Resnet12(1, 0.1)
    print(model(torch.randn(2,3,224,224)).shape)
    model = Resnet12(1, 0.1, use_fc=True)
    print(model(torch.randn(2,3,224,224)).shape)