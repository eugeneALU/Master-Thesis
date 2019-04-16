import torch.nn as nn

'''
Shallow and easy CNN model
    Input: 1 Channel image, (Aribitrary size actually)
'''

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False)



class ShallowCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ShallowCNN, self).__init__()

        self.conv1 = conv3x3(1, 16)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = conv3x3(16, 256)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = conv3x3(256, 1024)
        self.bn3 = nn.BatchNorm2d(1024)

        self.conv4 = conv3x3(1024, 1024)
        self.bn4 = nn.BatchNorm2d(1024)

        # self.conv5 = conv1x1(64, 32)
        # self.bn5 = nn.BatchNorm2d(32)

        # self.conv5 = conv1x1(64, 128)
        # self.bn5 = nn.BatchNorm2d(128)

        # self.conv6 = conv1x1(32, 16)
        # self.bn6 = nn.BatchNorm2d(16)

        # self.conv6 = conv1x1(128, 256)
        # self.bn6 = nn.BatchNorm2d(256)

        # self.fc = nn.Linear(16, 1) #output only one unit
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2= nn.Linear(4096, 1024)
        self.fc3= nn.Linear(1024, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.5)
        self.dropconv = nn.Dropout(p=0.2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv4(x)
        # x = self.bn4(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.conv5(x)
        # x = self.bn5(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.conv6(x)
        # x = self.bn6(x)
        # x = self.relu(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x

if __name__ == '__main__':
    from torchsummary import summary
    model = ShallowCNN()

    print("MODEL:")
    summary(model, input_size=(1, 96, 96), device="cpu")
    