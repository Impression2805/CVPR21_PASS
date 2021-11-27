import torch.nn as nn
import torch


class network(nn.Module):
    def __init__(self, numclass, feature_extractor):
        super(network, self).__init__()
        self.feature = feature_extractor
        self.fc = nn.Linear(512, numclass, bias=True)

    def forward(self, input):
        feature = self.feature(input)
        x = self.fc(feature)
        return x, feature

    def feature_extractor(self,inputs):
        return self.feature(inputs)

