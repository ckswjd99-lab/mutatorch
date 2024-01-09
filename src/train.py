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

from layers import MutableLinear, MutableConv2d
from utils import adjust_learning_rate, train, validate


# hyperparameters
EPOCHS = 50
BATCH_SIZE = 128
LEARNING_RATE = 0.05
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
WEIGHT_FORGET = 0.05
SEED = 42

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# data loading
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ]), download=True),
    batch_size=BATCH_SIZE, shuffle=True,
    num_workers=1, pin_memory=True)

val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=1, pin_memory=True)


# model definition
class MyMutableModel(nn.Module):
    def __init__(self):
        super(MyMutableModel, self).__init__()
        # input: Tensor[batch_size, 3, 32, 32]
        self.conv = nn.ModuleDict({
            'conv1_1': MutableConv2d(3, 16),
            'conv1_2': MutableConv2d(16, 16),
            'conv2_1': MutableConv2d(16, 32),
            'conv2_2': MutableConv2d(32, 32),
            'conv3_1': MutableConv2d(32, 64),
            'conv3_2': MutableConv2d(64, 64),
            'conv3_3': MutableConv2d(64, 64),
        })
        # self.fc = nn.ModuleDict({
        #     'fc1': MutableLinear(64 * 4 * 4, 1024),
        #     'fc2': MutableLinear(1024, 1024),
        #     'fc3': MutableLinear(1024, 10),
        # })
        self.fc = nn.ModuleDict({
            'fc1': nn.Linear(64 * 4 * 4, 1024),
            'fc2': nn.Linear(1024, 1024),
            'fc3': nn.Linear(1024, 10),
        })

        self.conv1_1 = self.conv['conv1_1']
        self.conv1_2 = self.conv['conv1_2']
        self.conv2_1 = self.conv['conv2_1']
        self.conv2_2 = self.conv['conv2_2']
        self.conv3_1 = self.conv['conv3_1']
        self.conv3_2 = self.conv['conv3_2']
        self.conv3_3 = self.conv['conv3_3']
        self.fc1 = self.fc['fc1']
        self.fc2 = self.fc['fc2']
        self.fc3 = self.fc['fc3']
        
        self.pool = nn.MaxPool2d(2, 2)

        self.model_config = [3, 16, 16, 32, 32, 64, 64, 64, 1024, 1024, 10]
        self.growth_size = [0, 2, 2, 4, 4, 8, 8, 0, 0, 0, 0]
        # self.growth_size = [0, 8, 8, 16, 16, 32, 32, 0, 512, 512, 0]

        self.mutable_layers = [
            self.conv1_1, self.conv1_2,
            self.conv2_1, self.conv2_2,
            self.conv3_1, self.conv3_2, self.conv3_3,
            self.fc1, self.fc2, self.fc3
        ]
    
    def forward(self, x):
        # input: Tensor[batch_size, 3, 32, 32]
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)

        # input: Tensor[batch_size, (16-), 16, 16]
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)

        # input: Tensor[batch_size, (32-), 8, 8]
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)

        # input: Tensor[batch_size, (64-), 4, 4]
        x = x.view(-1, self.model_config[7] * (4 * 4))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
    
    def grow_all(self):
        for i in range(len(self.mutable_layers)):
            self.model_config[i] += self.growth_size[i]
        
        self.conv1_1.modify_channels(self.model_config[0], self.model_config[1], weight_forget = WEIGHT_FORGET)
        self.conv1_2.modify_channels(self.model_config[1], self.model_config[2], weight_forget = WEIGHT_FORGET)
        self.conv2_1.modify_channels(self.model_config[2], self.model_config[3], weight_forget = WEIGHT_FORGET)
        self.conv2_2.modify_channels(self.model_config[3], self.model_config[4], weight_forget = WEIGHT_FORGET)
        self.conv3_1.modify_channels(self.model_config[4], self.model_config[5], weight_forget = WEIGHT_FORGET)
        self.conv3_2.modify_channels(self.model_config[5], self.model_config[6], weight_forget = WEIGHT_FORGET)
        self.conv3_3.modify_channels(self.model_config[6], self.model_config[7], weight_forget = WEIGHT_FORGET)
        # self.fc1.modify_features(self.model_config[7] * (4 * 4), self.model_config[8])
        # self.fc2.modify_features(self.model_config[8], self.model_config[9])
        # self.fc3.modify_features(self.model_config[9], self.model_config[10])

    
# model, criterion, optimizer
model = MyMutableModel().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)


# training
for epoch in range(EPOCHS):
    epoch_lr = adjust_learning_rate(optimizer, LEARNING_RATE, epoch, 10, 0.5)

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
            epoch, EPOCHS, lr=epoch_lr, train_acc=train_acc, train_loss=train_loss, val_acc=val_acc, val_loss=val_loss
        )
    )

    writer.add_scalar('train/accuracy', train_acc, epoch)
    writer.add_scalar('train/loss', train_loss, epoch)
    writer.add_scalar('val/accuracy', val_acc, epoch)
    writer.add_scalar('val/loss', val_loss, epoch)
    writer.add_scalar('lr', epoch_lr, epoch)

    if epoch in [4, 8, 16, 32, 48]:
        model.grow_all()
        print(
            'Model grows all layers:',
            model.model_config
        )
