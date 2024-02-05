import torch.nn as nn


# class Classifier(nn.Module):
#     def __init__(self):
#         super(Classifier, self).__init__()
#         # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
#         # torch.nn.MaxPool2d(kernel_size, stride, padding)
#         # input 维度 [1, 28, 28]
#         self.cnn = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),  # [10, 26, 26]
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),  # [64, 64, 64]
#
#             nn.Conv2d(64, 128, 3, 1, 1),  # [128, 64, 64]
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),  # [128, 32, 32]
#
#             nn.Conv2d(128, 256, 3, 1, 1),  # [256, 32, 32]
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),  # [256, 16, 16]
#
#             nn.Conv2d(256, 512, 3, 1, 1),  # [512, 16, 16]
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),  # [512, 8, 8]
#
#             nn.Conv2d(512, 512, 3, 1, 1),  # [512, 8, 8]
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2, 0),  # [512, 4, 4]
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(512 * 4 * 4, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 512),
#             nn.ReLU(),
#             nn.Linear(512, 11)
#         )
#
#     def forward(self, x):
#         out = self.cnn(x)
#         out = out.view(out.size()[0], -1)
#         return self.fc(out)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.Sequential(
            # The size of the picture is 28x28
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 14x14
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # The size of the picture is 7x7
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=7 * 7 * 64, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=10),
            nn.Softmax(dim=1)
        )

    def forward(self, input):
        output = self.model(input)
        return output

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        # 1*28*28
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(), nn.MaxPool2d(2, 2))  # 10*14*14
        self.conv2 = nn.Sequential(nn.Conv2d(10, 20, 3, 1, 1), nn.ReLU(),
                                   nn.MaxPool2d(2, 2))  #20*7*7
        self.conv3 = nn.Sequential(nn.Conv2d(20,30, 3, 1, 1),  nn.ReLU(),
                                   nn.MaxPool2d(2, 2))  # 30*3*3
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(30*3*3, 30),nn.Dropout(0.2), nn.Linear(30,10))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x