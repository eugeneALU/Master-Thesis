import torch.nn as nn
from collections import OrderedDict 

'''
Xception that has same architecture as the document.
    Input: 3 Channel image
'''

def depthwise_conv3x3(in_channel, out_channel, stride=1):
    return nn.Sequential(OrderedDict([
            ('depthwise_conv3x3', 
                nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=stride, padding=1, bias=False, groups=in_channel)),
            ('pointwise_conv1x1',
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False))
    ]))

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False)    

class Block(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, repeat=3, StartWithRelu=True, ResidualChannel_Up=False, WithMax=False):
        super(Block, self).__init__()
        
        layers = []
        for _ in range(repeat-1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(depthwise_conv3x3(in_channel,mid_channel))
            layers.append(nn.BatchNorm2d(mid_channel))

        layers.append(nn.ReLU(inplace=True))
        layers.append(depthwise_conv3x3(mid_channel,out_channel))
        layers.append(nn.BatchNorm2d(out_channel))
        
        if not StartWithRelu:
            layers = layers[1:]
        
        if WithMax:
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.ResidualChannel_Up = ResidualChannel_Up
        if self.ResidualChannel_Up:
            self.ChannelUp = nn.Sequential(
                    conv1x1(in_channel, out_channel, stride=2),
                    nn.BatchNorm2d(out_channel),
                )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        
        if self.ResidualChannel_Up:
            identity = self.ChannelUp(x)
        else:
            identity = x

        out = self.layers(x)

        out = out + identity

        return out

class Xception(nn.Module):
    def __init__(self, num_class=1, image_channel=3, FullMiddelBlock=True):
        super(Xception, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(image_channel,32,stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = conv3x3(32,64)
        self.bn2 = nn.BatchNorm2d(64)

        self.Entry1 = Block(64,128,128,repeat=2, StartWithRelu=False, ResidualChannel_Up=True, WithMax=True)
        self.Entry2 = Block(128,256,256,repeat=2, StartWithRelu=True, ResidualChannel_Up=True, WithMax=True)
        self.Entry3 = Block(256,728,728,repeat=2, StartWithRelu=True, ResidualChannel_Up=True, WithMax=True)

        self.Middel1 = Block(728,728,728)
        self.Middel2 = Block(728,728,728)
        self.Middel3 = Block(728,728,728)
        self.Middel4 = Block(728,728,728)
        if FullMiddelBlock:
            self.Middel5 = Block(728,728,728)
            self.Middel6 = Block(728,728,728)
            self.Middel7 = Block(728,728,728)
            self.Middel8 = Block(728,728,728)

        self.Exit = Block(728,728,1024,repeat=2, StartWithRelu=True, ResidualChannel_Up=True, WithMax=True)
        self.sep1 = depthwise_conv3x3(1024, 1536, stride=1)
        self.bn3 = nn.BatchNorm2d(1536)
        self.sep2 = depthwise_conv3x3(1536, 2048, stride=1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, num_class)
        # self.fc2 = nn.Linear(512, num_class)
        self.sigmoid = nn.Sigmoid()

        self.FullMiddelBlock = FullMiddelBlock

        # parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.Entry1(x)
        x = self.Entry2(x)
        x = self.Entry3(x)

        x = self.Middel1(x)
        x = self.Middel2(x)
        x = self.Middel3(x)
        x = self.Middel4(x)
        if self.FullMiddelBlock:
            x = self.Middel5(x)
            x = self.Middel6(x)
            x = self.Middel7(x)
            x = self.Middel8(x)

        x = self.Exit(x)
        x = self.sep1(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.sep2(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.sigmoid(x)

        return x

if __name__ == '__main__':
    from torchsummary import summary
    model = Xception()

    print("MODEL:")
    summary(model, input_size=(3, 384, 384), device="cpu")
    # for module in model.children():
    #     print(module)


