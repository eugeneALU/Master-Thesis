import os
import torch
import torch.optim
import pandas as pd
import numpy as np
from PIL import Image
from argparse import ArgumentParser
from Resnet_18 import resnet18_pretrain as MODEL
from PairLoss import ContrastiveLoss as LOSS
from torchvision import transforms
from torch.utils.data import DataLoader
from Pairdataset import PairDataset_threechannel as DATA
from tensorboardX import SummaryWriter
import torch.nn.functional as F


##########################
## PARSER
##########################
parser = ArgumentParser()
parser.add_argument("-n", "--network", choices=['Resnet_18'], default='Resnet_18', help='Specify the model')
parser.add_argument("-b", "--batch_size", type=int, choices=[16,32], default=32, help='Specify batch_size')
parser.add_argument("-s", "--size", type=int, choices=[224,299,384], default=224, help='Specify input image size')
parser.add_argument("-d", "--dataset", choices=['Image', 'MaskedImage'], default='Image', help='Specify dataset')
args = parser.parse_args()

path = os.path.join('log_Siamese', 'NEWDATA+aug_Siamese_ContrastiveLoss_Image_224', '10_checkpoint.pth')

param = torch.load(path)

model = MODEL(pretrain=False)
model.load_state_dict(param)
model.to('cuda')
model.eval()

if args.dataset == 'Image':
    SUFFIX = '_image.jpg'
else:
    SUFFIX = '_maskedimage.jpg'
SIZE = args.size
Batch_size = args.batch_size
path_to_testdata = os.path.join('..',args.dataset)
Test1 = os.path.join('..','PairLabel_test_MAXMAX.csv')
Test2 = os.path.join('..','PairLabel_test_minMAX.csv')
Test3 = os.path.join('..','PairLabel_test_minmin.csv')
Transform = transforms.Compose([
    transforms.Resize((SIZE,SIZE)),
    transforms.ToTensor()
])
LossFunction = LOSS()
path_to_testlabel = [Test1, Test2, Test3]

for path in path_to_testlabel:
    TEST = DATA(path_to_testdata, path, transform=Transform, image_suffix=SUFFIX)
    dataloader_TEST = DataLoader(TEST, Batch_size, num_workers=0)

    with torch.no_grad():
        Valid_loss = 0
        Valid_step = 0
        for _, (labels_test, img_test1, img_test2) in enumerate(dataloader_TEST):
            img_test1 = img_test1.cuda()
            img_test2 = img_test2.cuda()
            labels_test = labels_test.cuda()

            test_vector1 = model(img_test1)
            test_vector2 = model(img_test2)

            labels_test = labels_test.float()

            loss_test = LossFunction(test_vector1, test_vector2, labels_test)
            Valid_loss += loss_test.item()
            Valid_step += 1

        print(path, " Test loss: ", Valid_loss/Valid_step)

# loss for each label
for path in path_to_testlabel:
    test = pd.read_csv(path)
    test['DISTANCE'] = 0
    for index in range(len(test)):
        data = test.iloc[index]
        Image_name1 = data['PID1'] + SUFFIX
        Image_name2 = data['PID2'] + SUFFIX
        Stage1 = int(data['STAGE1'])
        Stage2 = int(data['STAGE2'])
        # label = data['LABEL']

        image1 = Image.open(os.path.join(path_to_testdata,str(Stage1),Image_name1)).convert('RGB')
        image2 = Image.open(os.path.join(path_to_testdata,str(Stage2),Image_name2)).convert('RGB')

        image1 = Transform(image1) 
        image2 = Transform(image2)
        image1 = image1.cuda()
        image2 = image2.cuda()
        # label = label.cuda()

        vector1 = model(image1.view(1,3,224,224))
        vector2 = model(image2.view(1,3,224,224))
        # label = label.float()
        # loss = LossFunction(vector1, vector2, label)
        distance = F.pairwise_distance(vector1, vector2)
        distance = distance.cpu()
    
        test.loc[index, 'DISTANCE'] = distance.detach().numpy()

    test.to_csv(os.path.join('..', '2_Result_'+path.split('_')[-1]), index=False)

# DummyInput = image1

model.cpu()
## add graph
with SummaryWriter(os.path.join('log_Siamese','graph_2')) as r:
    # DummyInput = DummyInput.cpu()
    # r.add_graph(model, DummyInput.view(1,3,224,224))

## add image
    # test = pd.read_csv(Test1)
    # for index in range(10):
    #     data = test.iloc[index]
    #     Image_name = data['PID2'] + SUFFIX
    #     Stage = str(int(data['STAGE2'])) 
    #     image = Image.open(os.path.join(path_to_testdata,Stage,Image_name))
    #     image = Transform(image) 
    #     r.add_image(Stage, image, global_step=0)

    # train = pd.read_csv(os.path.join('..','PairLabel_train_MAXMAX.csv'))
    # path_to_traindata = os.path.join('..',args.dataset)
    # for index in range(10):
    #     data = train.iloc[index]
    #     Image_name = data['PID2'] + SUFFIX
    #     Stage = str(int(data['STAGE2'])) 
    #     image = Image.open(os.path.join(path_to_traindata,Stage,Image_name))
    #     image = Transform(image) 
    #     r.add_image(Stage, image, global_step=1)
    
## add embedding
    test = pd.read_csv(Test1)
    meta = []
    mat = torch.zeros(0, 512) 
    label_img = torch.zeros(17, 3, SIZE, SIZE) 

    for index in range(17):
        data = test.iloc[index]
        Image_name = data['PID2'] + SUFFIX
        Stage = int(data['STAGE2']) 
        image = Image.open(os.path.join(path_to_testdata,str(Stage),Image_name)).convert('RGB')
        image = Transform(image) 
        vector = model(image.view(1,3,224,224))

        meta.append(Stage)
        mat = torch.cat((mat, vector.view(1,-1)), dim=0)
        label_img[index] = image 
    r.add_embedding(mat, metadata=meta, label_img=label_img, global_step=0)

    train = pd.read_csv(os.path.join('..','PairLabel_train_MAXMAX.csv'))
    path_to_traindata = os.path.join('..',args.dataset)

    meta = []
    mat = torch.zeros(0, 512) 
    label_img = torch.zeros(72, 3, SIZE, SIZE) 
    for index in range(72):
        data = train.iloc[index]
        Image_name = data['PID2'] + SUFFIX
        Stage = int(data['STAGE2'])
        image = Image.open(os.path.join(path_to_traindata,str(Stage),Image_name)).convert('RGB')
        image = Transform(image) 
        vector = model(image.view(1,3,224,224))

        meta.append(Stage)
        mat = torch.cat((mat, vector.view(1,-1)), dim=0)
        label_img[index] = image 
    
    r.add_embedding(mat, metadata=meta, label_img=label_img,global_step=1)