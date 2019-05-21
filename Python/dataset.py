import os
import torch
import pandas as pd
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
from random import randint

class MRIDataset(Dataset):
    def __init__(self, path_to_data, path_to_label, mode = 'R34', transform = transforms.ToTensor(), aug=False):
        self._mode = mode
        self._path = path_to_data
        self.transform = transform
        self.aug = aug

        self._label = pd.read_csv(path_to_label)
        self._label['label'] = 0
        if self._mode == 'R4':
            self._label.loc[self._label['stage'] > 3, 'label'] = 1
        elif self._mode == 'R34':
            self._label.loc[self._label['stage'] > 2, 'label'] = 1
        elif self._mode == 'R24':
            self._label.loc[self._label['stage'] > 1, 'label'] = 1
        elif self._mode == 'R14':
            self._label.loc[self._label['stage'] > 0, 'label'] = 1

        self.positive = self._label[self._label['label'] == 1].shape[0]
        self.negative = self._label[self._label['label'] == 0].shape[0]

    def __len__(self):
        return len(self._label)

    def __getitem__(self, index):
        data = self._label.iloc[index]
        Image_name = data['FileName']
        Stage = data['stage'] 
        label = data['label']

        image = Image.open(os.path.join(self._path,str(Stage),Image_name))

        if self.aug:
            I = randint(0,15)
            if I == 1:
                image = image.filter(ImageFilter.GaussianBlur(radius=2))
            elif I == 2:
                image = image.filter(ImageFilter.MedianFilter(size=3))
            elif I == 3:
                image = ImageEnhance.Contrast(image).enhance(2)
            elif I == 4:    
                image = ImageEnhance.Contrast(image).enhance(0.5)
            elif I == 5:    
                image = ImageEnhance.Brightness(image).enhance(2)
            elif I == 6:    
                image = ImageEnhance.Brightness(image).enhance(0.5)
            elif I == 7:    
                image = ImageEnhance.Sharpness(image).enhance(5)
            elif I == 8:
                image = image.transpose(Image.ROTATE_90)
            elif I == 9:
                image = image.transpose(Image.TRANSPOSE)
            elif I == 10:
                image = image.rotate(45)
            elif I == 11:
                image = image.rotate(135)
            elif I == 12:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif I == 13:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            elif I == 14:
                image = image.transpose(Image.ROTATE_180)
            elif I == 15:
                image = image.transpose(Image.ROTATE_270)

        if self.transform:
            image = self.transform(image) 

        return label, image

class MRIDataset_threechannel(Dataset):
    def __init__(self, path_to_data, path_to_label, mode = 'R34', transform = transforms.ToTensor(), aug=False):
        self._mode = mode
        self._path = path_to_data
        self.transform = transform
        self.aug = aug

        self._label = pd.read_csv(path_to_label)
        self._label['label'] = 0
        if self._mode == 'R4':
            self._label.loc[self._label['stage'] > 3, 'label'] = 1
        elif self._mode == 'R34':
            self._label.loc[self._label['stage'] > 2, 'label'] = 1
        elif self._mode == 'R24':
            self._label.loc[self._label['stage'] > 1, 'label'] = 1
        elif self._mode == 'R14':
            self._label.loc[self._label['stage'] > 0, 'label'] = 1

        self.positive = self._label[self._label['label'] == 1].shape[0]
        self.negative = self._label[self._label['label'] == 0].shape[0]

    def __len__(self):
        return len(self._label)

    def __getitem__(self, index):
        data = self._label.iloc[index]
        Image_name = data['FileName']
        Stage = data['stage'] 
        label = data['label']

        # read in as 3 channel image in shape (height, width, channel)
        image = Image.open(os.path.join(self._path,str(Stage),Image_name)).convert('RGB')

        if self.aug:
            I = randint(0,15)
            if I == 1:
                image = image.filter(ImageFilter.GaussianBlur(radius=2))
            elif I == 2:
                image = image.filter(ImageFilter.MedianFilter(size=3))
            elif I == 3:
                image = ImageEnhance.Contrast(image).enhance(2)
            elif I == 4:    
                image = ImageEnhance.Contrast(image).enhance(0.5)
            elif I == 5:    
                image = ImageEnhance.Brightness(image).enhance(2)
            elif I == 6:    
                image = ImageEnhance.Brightness(image).enhance(0.5)
            elif I == 7:    
                image = ImageEnhance.Sharpness(image).enhance(5)
            elif I == 8:
                image = image.transpose(Image.ROTATE_90)
            elif I == 9:
                image = image.transpose(Image.TRANSPOSE)
            elif I == 10:
                image = image.rotate(45)
            elif I == 11:
                image = image.rotate(135)
            elif I == 12:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            elif I == 13:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
            elif I == 14:
                image = image.transpose(Image.ROTATE_180)
            elif I == 15:
                image = image.transpose(Image.ROTATE_270)

        if self.transform:
            # transfroms.ToTensor will change shape into (channel. height, width)
            image = self.transform(image) 

        return label, image

if __name__ == '__main__':
    from torch.utils.data import DataLoader, WeightedRandomSampler

    path_to_data = '../Image'
    path_to_label = '../Label_train_area2500.csv'
    Transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    # Data = MRIDataset(path_to_data, path_to_label,transform=Transform)
    # weight = 1. / torch.tensor([Data.negative,Data.positive], dtype=torch.float)
    # target = torch.tensor(Data._label['label'], dtype=torch.long)
    # sample_weight = torch.tensor([weight[t] for t in target], dtype=torch.float)
    # sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
    # print("length of data = ",len(Data))
    # # print(sample_weight)
    # print(Data.negative,Data.positive)
    # print(len(Data._label[Data._label['label'] == 1]['stage']))
    # dataloader = DataLoader(Data, 32, sampler=sampler, num_workers=0, drop_last=True)

    Data = MRIDataset_threechannel(path_to_data, path_to_label,transform=Transform)
    weight = 1. / torch.tensor([Data.negative,Data.positive], dtype=torch.float)
    target = torch.tensor(Data._label['label'], dtype=torch.long)
    sample_weight = torch.tensor([weight[t] for t in target], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
    dataloader = DataLoader(Data, 32, num_workers=0, drop_last=True, sampler=sampler)
    print("length of data = ",len(Data))
    # print(sample_weight)
    print(Data.negative,Data.positive)
    print(len(Data._label[Data._label['label'] == 1]['stage']))

    '''check the image and label data''' 
    for batch_index, (labels, img) in enumerate(dataloader):
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