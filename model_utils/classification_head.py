import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes, n_head_layers, drop_path_rate):
        super(ClassificationHead, self).__init__()
        self.n_head_layers = n_head_layers
        self.drop_path_rate = drop_path_rate

        layers = []
        for i in range(self.n_head_layers):
            if i == self.n_head_layers-1:
                layers.append(nn.Linear(in_features, num_classes))
            elif i == 0:
                #only add batchnorm and dropout for the first layer if there are more than 1 head layers
                layers.append(nn.BatchNorm1d(in_features))
                layers.append(nn.Linear(in_features, in_features))
                layers.append(nn.LeakyReLU())
                layers.append(nn.Dropout(p=self.drop_path_rate))
            else:
                layers.append(nn.Linear(in_features, in_features))
                layers.append(nn.LeakyReLU())

        self.head_layers = nn.Sequential(*layers)

    def forward(self, x):
        for layer in self.head_layers:
            x = layer(x)
        return x