import os
import torch
import torch.optim
from dataset import MRIDataset_threechannel
from Resnet_pretrain import resnet50_pretrain
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.nn.functional import binary_cross_entropy

path_to_logs_dir = './log/resnet_pretrain'
path_to_data = '../Image_train'
path_to_label = '../Label_train.csv'
path_to_testdata = '../Image_test'
path_to_testlabel = '../Label_test.csv'
Batch_size = 32
Batch_size_test = 512
Epochs = 5
Transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor()
])

if __name__ == '__main__':
    writer = SummaryWriter(path_to_logs_dir)
    dataset = MRIDataset_threechannel(path_to_data, path_to_label, mode='R34', transform=Transform)
    weight = 1. / torch.tensor([dataset.negative,dataset.positive], dtype=torch.float)
    target = torch.tensor(dataset._label['label'], dtype=torch.long)
    sample_weight = torch.tensor([weight[t] for t in target], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
    dataloader = DataLoader(dataset, Batch_size, sampler=sampler, num_workers=1, drop_last=True)

    dataset_test = MRIDataset_threechannel(path_to_testdata, path_to_testlabel, mode='R34', transform=Transform)
    dataloader_test = DataLoader(dataset_test, Batch_size_test, shuffle=True, num_workers=0, drop_last=True)

    resnet = resnet50_pretrain()
    # resnet = resnet.cuda()
    resnet.train(mode=True)
    optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-3)
    epoch = 0
    step = 1

    while epoch != Epochs:
        for batch_index, (labels, img) in enumerate(dataloader):
            # img = img.cuda()
            # labels = labels.cuda()
            logits = resnet(img)
            logits = logits.squeeze(1)
            labels = labels.float()
            loss = binary_cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print('epoch: {} step: {} loss: {}' .format(epoch, step, loss.item()))

            if step % 20 == 0:
                writer.add_scalar('train/loss', loss.item(), step)
                resnet.eval()

                with torch.no_grad():
                    Valid_loss = 0
                    Valid_step = 0
                    for _, (labels_test, img_test) in enumerate(dataloader_test):
                    # img_test = img_test.cuda()
                    # labels_test = labels_test.cuda()
                        logits_test = resnet(img_test)
                        logits_test = logits_test.squeeze(1)
                        labels_test = labels_test.float()
                        loss_test = binary_cross_entropy(logits_test, labels_test)
                        Valid_loss += loss_test.item()
                        Valid_step += 1

                    writer.add_scalar('valid/loss', Valid_loss/Valid_step , step)

                resnet.train(mode=True)
            step += 1
        epoch += 1
        torch.save(resnet.state_dict(), os.path.join(path_to_logs_dir, 'checkpoint.pth'))
    torch.save(resnet.state_dict(), os.path.join(path_to_logs_dir, 'parameter.pth'))