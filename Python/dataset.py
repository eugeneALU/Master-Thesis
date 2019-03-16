import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MRIDataset(Dataset):
    def __init__(self, path_to_data, path_to_label, mode = 'R34', transform = transforms.ToTensor()):
        self._mode = mode
        self._path = path_to_data
        self.transform = transform

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

        if self.transform:
            image = self.transform(image) 

        return label, image

class MRIDataset_threechannel(Dataset):
    def __init__(self, path_to_data, path_to_label, mode = 'R34', transform = transforms.ToTensor()):
        self._mode = mode
        self._path = path_to_data
        self.transform = transform

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
        image = Image.open(os.path.join(self._path,str(Stage),Image_name)).mode('RGB')

        if self.transform:
            # transfroms.ToTensor will change shape into (channel. height, width)
            image = self.transform(image) 

        return label, image

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    path_to_data = '../Image_train'
    path_to_label = '../Label_train.csv'
    Data = MRIDataset(path_to_data, path_to_label,transform=transforms.ToTensor())
    weight = 1. / torch.tensor([Data.negative,Data.positive], dtype=torch.float)
    target = torch.tensor(Data._label['label'], dtype=torch.long)
    sample_weight = torch.tensor([weight[t] for t in target], dtype=torch.float)
    print("length of data = ",len(Data))
    print(sample_weight)
    print(Data.negative,Data.positive)
    print(len(Data._label[Data._label['label'] == 1]['stage']))

    Data = MRIDataset_threechannel(path_to_data, path_to_label,transform=transforms.ToTensor())
    weight = 1. / torch.tensor([Data.negative,Data.positive], dtype=torch.float)
    target = torch.tensor(Data._label['label'], dtype=torch.long)
    sample_weight = torch.tensor([weight[t] for t in target], dtype=torch.float)
    print("length of data = ",len(Data))
    print(sample_weight)
    print(Data.negative,Data.positive)
    print(len(Data._label[Data._label['label'] == 1]['stage']))

    '''check the image and label data''' 
    #for label, image in Data:
    #    pass
    #print(image)
    #print('MAX: ', image.max())
    #print('MIN: ', image.min())