import torch.nn as nn

def conv3x3(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channel, out_channel, stride=1):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, padding=0, bias=False)

class Bottleneck(nn.Module):
    # bind to class itself(before instantiation)
    expansion = 4
    def __init__(self, in_channel, channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        
        self.stride = stride

        self.conv1 = conv1x1(in_channel, channel)
        self.bn1 = nn.BatchNorm2d(channel)
        self.conv2 = conv3x3(channel, channel, stride)
        self.bn2 = nn.BatchNorm2d(channel)
        self.conv3 = conv1x1(channel, channel * self.expansion)
        self.bn3 = nn.BatchNorm2d(channel * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out 

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=2, zero_init_residual=False):    
        super(ResNet, self).__init__()
        self.identity_channel = 64

        # input channel is 1 here, since we are gary-scale image
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1) #output only one unit 
        self.sigmoid = nn.Sigmoid()

        # parameter initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, in_channel, blocks, stride=1):
        downsample = None
        out_channel = in_channel * block.expansion
        # stride != 1 : image size change -> downsampling of identity copy
        # identity_channel != out_channel : channel not match -> conv1x1 to change the channel of identity copy
        if stride != 1 or self.identity_channel != out_channel:
            downsample = nn.Sequential(
                conv1x1(self.identity_channel, out_channel, stride),
                nn.BatchNorm2d(out_channel),
            )

        # list to store layers
        layers = []
        # first block handles here, need downsample
        layers.append(block(self.identity_channel, in_channel, stride, downsample)) 
        # after first block, identity has the same channel as the out_channel/ and don't need donwsample anymore
        self.identity_channel = out_channel
        for _ in range(1, blocks):      # from 1,2,.... to blocks-1
            layers.append(block(self.identity_channel, in_channel))

        # create layers by unpacked the "layers" list
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)

        return x    
    
def resnet50(**kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    return model


if __name__ == '__main__':
    model = resnet50()

    print("MODEL:")
    for module in model.children():
        print(module)
