import os
import torch
import torch.optim
from argparse import ArgumentParser
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tensorboardX import SummaryWriter
from torch.nn.functional import binary_cross_entropy
from sklearn.metrics import accuracy_score

##########################
## PARSER
##########################
parser = ArgumentParser()
parser.add_argument("-n", "--network", choices=['Resnet_onechannel', 'Resnet_pretrain', 'Resnet_18', 'Resnet_50', 'Resnet_101'
                    'Resnet_depthwise', 'Xception'], default='Resnet_50', help='Specify the model')
parser.add_argument("-c", "--channel", type=int, choices=[1,3], default=3, help='Specify the number of input image')
parser.add_argument("-b", "--batch_size", type=int, choices=[16,32], default=32, help='Specify batch_size')
parser.add_argument("-e", "--epochs", type=int, default=30, help='Specify epochs')
parser.add_argument("-s", "--size", type=int, choices=[299,384,224], default=224, help='Specify input image size')
parser.add_argument("-m", "--mode", choices=['R24','R34','R4'], default='R34', help='Specify classify mode')
parser.add_argument("-d", "--dataset", choices=['Image','MaskedImage'], default='MaskedImage', help='Specify the dataset')

args = parser.parse_args()
path_to_logs_dir = os.path.join('.','log_Resnet18','ResNet50_ImageSize_'+str(args.size)+'_'+args.dataset+'_area10000_aug2(1e-5)_weightdecay(1)')
path_to_data = os.path.join('..',args.dataset)
path_to_testdata = os.path.join('..',args.dataset)

if args.dataset == 'Image':
    path_to_label = '../Label_train_area10000.csv'
    path_to_testlabel = '../Label_valid_area10000.csv'
else:
    path_to_label = '../MaskedLabel_train_area10000.csv'
    # path_to_label = '../MaskedLabel_train_all_area10000.csv'
    path_to_testlabel = '../MaskedLabel_valid_area10000.csv' 

Batch_size = args.batch_size
EPOCHS = args.epochs
SIZE = args.size
MODE = args.mode
if args.channel == 1:
    from dataset import MRIDataset as DATA
else:
    from dataset import MRIDataset_threechannel as DATA

if args.network == 'Resnet_onechannel':
    from Resnet_onechannel import resnet50_onechannel as MODEL
elif args.network == 'Resnet_pretrain':
    from Resnet_pretrain import resnet50_pretrain as MODEL
elif args.network == 'Resnet_depthwise':
    from Resnet_depthwise import resnet50_depthwise as MODEL
elif args.network == 'Xception':
    from Xception import Xception as MODEL
elif args.network == 'Resnet_18':
    from Resnet import resnet18_pretrain as MODEL
elif args.network == 'Resnet_50':
    from Resnet import resnet50_pretrain as MODEL
elif args.network == 'Resnet_101':
    from Resnet import resnet101_pretrain as MODEL

Transform = transforms.Compose([
    transforms.Resize((SIZE,SIZE)),
    transforms.ToTensor()
])

if __name__ == '__main__':
    # writer = SummaryWriter(path_to_logs_dir)
    dataset = DATA(path_to_data, path_to_label, mode=MODE, transform=Transform, aug=True)
    weight = 1. / torch.tensor([dataset.negative,dataset.positive], dtype=torch.float)
    target = torch.tensor(dataset._label['label'], dtype=torch.long)
    sample_weight = torch.tensor([weight[t] for t in target], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight))
    dataloader = DataLoader(dataset, Batch_size, sampler=sampler, num_workers=1, drop_last=True)

    dataset_test = DATA(path_to_testdata, path_to_testlabel, mode=MODE, transform=Transform, aug=False)
    dataloader_test = DataLoader(dataset_test, Batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = MODEL()
    model = model.cuda()
    model.train(mode=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1) #, weight_decay=0.1
    epoch = 1
    step = 1

    while epoch <= EPOCHS:
        for batch_index, (labels, img) in enumerate(dataloader):
            img = img.cuda()
            labels = labels.cuda()
            logits = model(img)
            logits = logits.squeeze(1)
            labels = labels.float()
            loss = binary_cross_entropy(logits, labels)
            logits = logits > 0.5
            accu = accuracy_score(logits.cpu(),labels.cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print('epoch: {} step: {} loss: {}' .format(epoch, step, loss.item()))

            # if step % 20 == 0:
            #     writer.add_scalar('train/loss', loss.item(), step)
            #     writer.add_scalar('train/accu', accu, step)
            #     model.eval()

            #     with torch.no_grad():
            #         Valid_loss = 0
            #         Valid_step = 0
            #         Valid_accu = 0
            #         for _, (labels_test, img_test) in enumerate(dataloader_test):
            #             img_test = img_test.cuda()
            #             labels_test = labels_test.cuda()
            #             logits_test = model(img_test)
            #             logits_test = logits_test.squeeze(1)
            #             labels_test = labels_test.float()
            #             loss_test = binary_cross_entropy(logits_test, labels_test)
            #             logits_test = logits_test > 0.5
            #             Valid_loss += loss_test.item()
            #             Valid_accu += accuracy_score(logits_test.cpu(),labels_test.cpu())
            #             Valid_step += 1

            #         writer.add_scalar('valid/loss', Valid_loss/Valid_step , step)
            #         writer.add_scalar('valid/accu', Valid_accu/Valid_step , step)

            #     model.train(mode=True)
            step += 1
        if epoch%5==0:
            torch.save(model.state_dict(), os.path.join(path_to_logs_dir, 'checkpoint'+str(epoch)+'.pth'))
        epoch += 1
    torch.save(model.state_dict(), os.path.join(path_to_logs_dir, 'parameter.pth'))