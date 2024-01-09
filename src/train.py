import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

from tqdm import tqdm

from utils import adjust_learning_rate, train, validate
from layers import MutableLinear, MutableConv2d
from models import MyMutableVGG


# hyperparameters
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
WEIGHT_FORGET = 0.0
SEED = 42

# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False


# data loading
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=1, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=1, pin_memory=True)

    
# model, criterion, optimizer
model = MyMutableVGG().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


# training
for epoch in range(EPOCHS):
    epoch_lr = adjust_learning_rate(optimizer, LEARNING_RATE, epoch, 20, 0.5)

    # train for one epoch
    train_acc, train_loss = train(train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    val_acc, val_loss = validate(val_loader, model, criterion, epoch)

    # print training/validation statistics
    print(
        'Epoch: {0}/{1}\t'
        'LR: {lr:.6f}\t'
        'Train Accuracy {train_acc:.3f}\t'
        'Train Loss {train_loss:.3f}\t'
        'Val Accuracy {val_acc:.3f}\t'
        'Val Loss {val_loss:.3f}'
        .format(
            epoch + 1, EPOCHS, lr=epoch_lr, train_acc=train_acc, train_loss=train_loss, val_acc=val_acc, val_loss=val_loss
        )
    )

    writer.add_scalar('train/accuracy', train_acc, epoch)
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('val/accuracy', val_acc, epoch)
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('lr', epoch_lr, epoch)

    if epoch in [4, 8, 16, 32, 48]:
        rand_input = torch.rand(1, 3, 32, 32).cuda()
        model.grow_all(weight_forget=WEIGHT_FORGET)
        optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
        
        print(
            'Model grows all layers:',
            model.model_config
        )
