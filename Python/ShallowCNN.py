import torch.nn as nn

'''
Shallow and easy CNN model
    Input: 1 Channel image, (Aribitrary size actually)
'''

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=True)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=True)



class ShallowCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(ShallowCNN, self).__init__()

        # self.bn = nn.BatchNorm2d(1)
        self.conv1 = conv3x3(1, 6)
        self.bn1 = nn.BatchNorm2d(6)

        self.conv2 = conv3x3(6, 6)
        self.bn2 = nn.BatchNorm2d(6)

        self.conv3 = conv3x3(6,16)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = conv3x3(16,16)
        self.bn4 = nn.BatchNorm2d(16)

        self.conv5 = conv1x1(16, 32)
        self.bn5 = nn.BatchNorm2d(32)

        # self.conv5 = conv1x1(64, 128)
        # self.bn5 = nn.BatchNorm2d(128)

        # self.conv6 = conv1x1(32, 16)
        # self.bn6 = nn.BatchNorm2d(16)

        self.conv6 = conv1x1(32, 32)
        self.bn6 = nn.BatchNorm2d(32)

        # self.fc = nn.Linear(16, 1) #output only one unit
        self.fc1 = nn.Linear(32, 1)
        # self.fc2= nn.Linear(512, 256)
        # self.fc3= nn.Linear(256, 1)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(p=0.5)
        self.dropconv = nn.Dropout(p=0.2)

        # parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        # x = self.bn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.maxpool(x)
        x = self.relu(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1)
        # x = self.drop(x)
        x = self.fc1(x)
        # x = self.drop(x)
        # x = self.fc2(x)
        # x = self.drop(x)
        # x = self.fc3(x)
        x = self.sigmoid(x)

        return x

if __name__ == '__main__':
    from torchsummary import summary
    model = ShallowCNN()

    print("MODEL:")
    summary(model, input_size=(1, 384, 384), device="cpu")
    