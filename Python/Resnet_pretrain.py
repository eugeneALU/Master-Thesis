import torch.nn as nn
from torchvision import models

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

    def forward(self, x):
        x = self.model(x)
        x = self.sigmoid(x)

        return x        

if __name__ == '__main__':
    model = resnet50_pretrain()

    print("MODEL:")
    for module in model.children():
        print(module)