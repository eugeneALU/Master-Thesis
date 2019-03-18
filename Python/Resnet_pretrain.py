import torch.nn as nn
from torchvision import models

'''
Pretrained Resnet, include from official module 
    Input: 3 Channel image, (Aribitrary size actually)
'''

class resnet50_pretrain(nn.Module):
    def __init__(self, num_class=2, freeze=False):
        super(resnet50_pretrain, self).__init__()

        self.num_class = num_class
        self.model = models.resnet50(pretrained=True)
        self.sigmoid = nn.Sigmoid()

        # freeze the model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)
        self.model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)

        return x        

if __name__ == '__main__':
    from torchsummary import summary
    model = resnet50_pretrain()

    print("MODEL:")
    summary(model, input_size=(3, 384, 384), device="cpu")
    # for module in model.children():
        # print(module)