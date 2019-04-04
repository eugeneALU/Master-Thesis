import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision.models.inception import Inception3

'''
Pretrained Inception_v3, include from official module
    Input: 3 Channel image, (Aribitrary size actually)
'''
URL = 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth'

class inception_v3_pretrain(nn.Module):
    def __init__(self, num_class=2, freeze=False, pretrain=True):
        super(inception_v3_pretrain, self).__init__()

        self.num_class = num_class
        self.model = Inception3(aux_logits=True, transform_input=False)
        if pretrain:
            self.model.load_state_dict(model_zoo.load_url(URL))

        self.sigmoid = nn.Sigmoid()

        # freeze the model
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

        in_features = self.model.fc.in_features
        in_features_aux = self.model.AuxLogits.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)
        self.model.AuxLogits.fc = nn.Linear(in_features_aux, 1)
        #self.fc2 = nn.Linear(512,1)

    def forward(self, x):
        if self.model.training and self.model.aux_logits:
            x, aux_x = self.model(x)
            aux_x = self.sigmoid(aux_x)
            #x = self.fc2(x)
            x = self.sigmoid(x)
            return x, aux_x 
        else:
            x = self.model(x)
            #x = self.fc2(x)
            x = self.sigmoid(x)
            return x    


if __name__ == '__main__':
    from torchsummary import summary
    model = inception_v3_pretrain()

    print("MODEL:")
    summary(model, input_size=(3, 299,299), device="cpu")
    # for module in model.children():
        # print(module)
    # for param in model.parameters():
    #     print(param.requires_grad)
    #print(model.state_dict() )