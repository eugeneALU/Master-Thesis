import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from random import randint

class PairDataset(Dataset):
    def __init__(self, path_to_data, path_to_label, transform = transforms.ToTensor(), image_suffix = '_image.jpg'):
        self._path = path_to_data
        self.transform = transform
        self.suffix = image_suffix

        self._label = pd.read_csv(path_to_label)

        self.positive = self._label[self._label['LABEL'] == 1].shape[0]
        self.negative = self._label[self._label['LABEL'] == 0].shape[0]

    def __len__(self):
        return len(self._label)

    def __getitem__(self, index):
        data = self._label.iloc[index]
        Image_name1 = data['PID1'] + self.suffix
        Image_name2 = data['PID2'] + self.suffix
        Stage1 =  int(data['STAGE1'])
        Stage2 =  int(data['STAGE2'])
        label = data['LABEL']

        image1 = Image.open(os.path.join(self._path,str(Stage1),Image_name1))
        image2 = Image.open(os.path.join(self._path,str(Stage2),Image_name2))

        if self.transform:
            image1 = self.transform(image1) 
            image2 = self.transform(image2) 

        return label, image1, image2

class PairDataset_threechannel(Dataset):
    def __init__(self, path_to_data, path_to_label, transform = transforms.ToTensor(), image_suffix = '_image.jpg', aug=True):
        self._path = path_to_data
        self.transform = transform
        self.suffix = image_suffix
        self.aug = aug

        self._label = pd.read_csv(path_to_label)

        self.positive = self._label[self._label['LABEL'] == 1].shape[0]
        self.negative = self._label[self._label['LABEL'] == 0].shape[0]

    def __len__(self):
        return len(self._label)

    def __getitem__(self, index):
        data = self._label.iloc[index]
        Image_name1 = data['PID1'] + self.suffix
        Image_name2 = data['PID2'] + self.suffix
        Stage1 = int(data['STAGE1'])
        Stage2 = int(data['STAGE2']) 
        label = data['LABEL']

        # read in as 3 channel image in shape (height, width, channel)
        image1 = Image.open(os.path.join(self._path,str(Stage1),Image_name1)).convert('RGB')
        image2 = Image.open(os.path.join(self._path,str(Stage2),Image_name2)).convert('RGB')
        
        if self.aug:
            I = randint(0,8)
            if I == 1:
                image1 = image1.transpose(Image.ROTATE_90)
                image2 = image2.transpose(Image.ROTATE_90)
            elif I == 2:
                image1 = image1.transpose(Image.TRANSPOSE)
                image2 = image2.transpose(Image.TRANSPOSE)
            elif I == 3:
                image1 = image1.rotate(45)
                image2 = image2.rotate(45)
            elif I == 4:    
                image1 = image1.rotate(135)
                image2 = image2.rotate(135)
            elif I == 5:    
                image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
                image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
            elif I == 6:    
                image1 = image1.transpose(Image.FLIP_TOP_BOTTOM)
                image2 = image2.transpose(Image.FLIP_TOP_BOTTOM)
            elif I == 7:    
                image1 = image1.transpose(Image.ROTATE_180)
                image2 = image2.transpose(Image.ROTATE_180)
            elif I == 8:
                image1 = image1.transpose(Image.ROTATE_270)
                image2 = image2.transpose(Image.ROTATE_270)

        if self.transform:
            # transfroms.ToTensor will change shape into (channel. height, width)
            image1 = self.transform(image1) 
            image2 = self.transform(image2) 

        return label, image1, image2

if __name__ == '__main__':
    from torch.utils.data import DataLoader, WeightedRandomSampler

    path_to_data = '../Image_train'
    path_to_label = '../PairLabel_train.csv'
    Transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # Data = PairDataset(path_to_data, path_to_label,transform=Transform)
    # weight = 1. / torch.tensor([Data.negative,Data.positive], dtype=torch.float)
    # target = torch.tensor(Data._label['LABEL'], dtype=torch.long)
    # sample_weight = torch.tensor([weight[t] for t in target], dtype=torch.float)
    # sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
    # print("length of data = ",len(Data))
    # # print(sample_weight)
    # print(Data.negative,Data.positive)
    # print(len(Data._label[Data._label['LABEL] == 1]))
    # dataloader = DataLoader(Data, 32, sampler=sampler, num_workers=0, drop_last=True)

    Data = PairDataset_threechannel(path_to_data, path_to_label,transform=Transform)
    weight = 1. / torch.tensor([Data.negative,Data.positive], dtype=torch.float)
    target = torch.tensor(Data._label['LABEL'], dtype=torch.long)
    sample_weight = torch.tensor([weight[t] for t in target], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
    dataloader = DataLoader(Data, 32, num_workers=0, drop_last=True, sampler=sampler)
    print("length of data = ",len(Data))
    # print(sample_weight)
    print(Data.negative,Data.positive)
    print(len(Data._label[Data._label['LABEL'] == 1]))

    '''check the image and label data'''
    for batch_index, (labels, img1, img2) in enumerate(dataloader):
        ones = 0 
        zeros = 0 
        for label in labels:
            if label == 1:
                ones += 1
            else:
                zeros += 1
        print(batch_index)
        print("ones: ", ones)
        print("zeros: ", zeros)