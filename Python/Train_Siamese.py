import os
import torch
import torch.optim
from argparse import ArgumentParser
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from tensorboardX import SummaryWriter
from PairLoss import ContrastiveLoss as LOSS

##########################
## PARSER
##########################
parser = ArgumentParser()
parser.add_argument("-n", "--network", choices=['Resnet_18'], default='Resnet_18', help='Specify the model')
parser.add_argument("-c", "--channel", type=int, choices=[1,3], default=3, help='Specify the number of input image')
parser.add_argument("-b", "--batch_size", type=int, choices=[16,32], default=32, help='Specify batch_size')
parser.add_argument("-e", "--epochs", type=int, default=20, help='Specify epochs')
parser.add_argument("-s", "--size", type=int, choices=[224,299,384], default=224, help='Specify input image size')
parser.add_argument("-d", "--dataset", choices=['Image', 'MaskedImage'], default='Image', help='Specify dataset')

args = parser.parse_args()
path_to_logs_dir = os.path.join('.','log_Siamese','NEWDATA+aug_Siamese_ContrastiveLoss_'+args.dataset+'_'+str(args.size))
path_to_data = os.path.join('..',args.dataset)
path_to_testdata = os.path.join('..',args.dataset)
path_to_label = '../PairLabel_train.csv'
path_to_testlabel = '../PairLabel_valid.csv'


Batch_size = args.batch_size
EPOCHS = args.epochs
SIZE = args.size

if args.dataset == 'Image':
    SUFFIX = '_image.jpg'
else:
    SUFFIX = '_maskedimage.jpg'

if args.channel == 1:
    from Pairdataset import PairDataset as DATA
else:
    from Pairdataset import PairDataset_threechannel as DATA

if args.network == 'Resnet_18':
    from Resnet_18 import resnet18_pretrain as MODEL
# elif args.network == 'Resnet_pretrain':
#     from Resnet_pretrain import resnet50_pretrain as MODEL
# elif args.network == 'Resnet_depthwise':
#     from Resnet_depthwise import resnet50_depthwise as MODEL
# elif args.network == 'Xception':
#     from Xception import Xception as MODEL
# elif args.network == 'Inception_v3':
#     from Inception_v3 import inception_v3_pretrain as MODEL

Transform = transforms.Compose([
    transforms.Resize((SIZE,SIZE)),
    transforms.ToTensor()
])

if __name__ == '__main__':
    writer = SummaryWriter(path_to_logs_dir)
    dataset = DATA(path_to_data, path_to_label,transform=Transform, image_suffix=SUFFIX, aug=True)

    weight = 1. / torch.tensor([dataset.negative,dataset.positive], dtype=torch.float)
    target = torch.tensor(dataset._label['LABEL'], dtype=torch.long)
    sample_weight = torch.tensor([weight[t].item() for t in target], dtype=torch.float)
    sampler = WeightedRandomSampler(sample_weight, len(sample_weight))

    dataloader = DataLoader(dataset, Batch_size, sampler=sampler, num_workers=1, drop_last=True)

    dataset_test = DATA(path_to_testdata, path_to_testlabel,transform=Transform, image_suffix=SUFFIX, aug=True)
    dataloader_test = DataLoader(dataset_test, Batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = MODEL()
    model = model.cuda()
    
    model.train(mode=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    LossFunction = LOSS()
    epoch = 0
    step = 1

    while epoch != EPOCHS:
        for batch_index, (labels, img1, img2) in enumerate(dataloader):
            img1 = img1.cuda()
            img2 = img2.cuda()
            labels = labels.cuda()
            
            vector1 = model(img1)
            vector2 = model(img2)

            labels = labels.float()

            loss = LossFunction(vector1, vector2, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 10 == 0:
                print('epoch: {} step: {} loss: {}' .format(epoch, step, loss.item()))

            if step % 20 == 0:
                writer.add_scalar('train/loss', loss.item(), step)
                model.eval()

                with torch.no_grad():
                    Valid_loss = 0
                    Valid_step = 0
                    for _, (labels_test, img_test1, img_test2) in enumerate(dataloader_test):
                        img_test1 = img_test1.cuda()
                        img_test2 = img_test2.cuda()
                        labels_test = labels_test.cuda()

                        test_vector1 = model(img_test1)
                        test_vector2 = model(img_test2)

                        labels_test = labels_test.float()

                        loss_test = LossFunction(test_vector1, test_vector2, labels_test)
                        Valid_loss += loss_test.item()
                        Valid_step += 1

                    writer.add_scalar('valid/loss', Valid_loss/Valid_step , step)

                model.train(mode=True)
            step += 1
        epoch += 1
        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(path_to_logs_dir, str(epoch)+'_checkpoint.pth'))
    torch.save(model.state_dict(), os.path.join(path_to_logs_dir, 'parameter.pth'))
    
    writer.close() 