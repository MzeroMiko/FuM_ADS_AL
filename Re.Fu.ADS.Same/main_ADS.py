import random, time, os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100, CIFAR10

import config
from resnet_cifar import BasicBlock, Bottleneck, ResNet as ResNet_Cifar


class printlog():
    def __init__(self, filename, sync_terminal=True, mode='w+'):
        folder = os.path.split(os.path.realpath(filename))[0]
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.file = open(filename, mode=mode)
        self.sync_terminal = sync_terminal
        self.gettime = lambda:time.strftime('%Y/%m/%d %H:%M:%S', time.localtime())

    def log(self, *x, withtime=True):
        timestr = f'[{self.gettime()}] =>' if withtime else ''
        print(timestr, *x, file=self.file, flush=True)
        if self.sync_terminal:
            print(timestr, *x)

    def close(self):
        self.file.close()


class SubsetSequentialSampler(torch.utils.data.Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))
    
    def __len__(self):
        return len(self.indices)


def get_dataset(dataset='cifar10', root=''):

    cifar = {'cifar10': CIFAR10, 'cifar100': CIFAR100}[dataset]
    mean = {'cifar10': [0.4914, 0.4822, 0.4465], 'cifar100': [0.5071, 0.4867, 0.4408]}[dataset]
    std = {'cifar10': [0.2023, 0.1994, 0.2010], 'cifar100': [0.2675, 0.2565, 0.2761]}[dataset]

    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    train = cifar(root, train=True, download=True, transform=train_transform)
    unlabeled = cifar(root, train=True, download=True, transform=train_transform)
    test = cifar(root, train=False, download=True, transform=test_transform)
    
    return train, test, unlabeled


class metric_entropy(nn.Module):
    # 286.FuM-ADS: Fomular 3
    def __init__(self):
        super(metric_entropy, self).__init__()

    def forward(self, scores1, scores2):
        # normalization: (as train_ADS.py: line 12), but not in 286.FuM.pdf??
        # scores1 = scores1 / torch.sum(scores1, dim=1)[None, scores1.size(1)] # wrong!!
        # scores2 = scores2 / torch.sum(scores2, dim=1)[None, scores1.size(1)] # wrong!!
        scores1 = scores1 / torch.sum(scores1, dim=1).unsqueeze(dim=1).repeat(1, scores1.size(1))
        scores2 = scores2 / torch.sum(scores2, dim=1).unsqueeze(dim=1).repeat(1, scores2.size(1))
        return -torch.sum(scores1*torch.log2(scores1) + scores2*torch.log2(scores2), dim=1) / 2


class criterion_init(nn.Module):
    # 286.FuM-ADS: Fomular 1
    # train_ADS.py: line33-37
    def __init__(self, reduction='none'):
        super(criterion_init, self).__init__()
        self.criterion = nn.BCELoss(reduction=reduction)

    def forward(self, scores1, scores2, labels):
        # scoresN: scores tensor given by different nets with the same data batch; 
        # labels: labels of this data batch;
        return torch.sum(self.criterion(scores1, labels) + self.criterion(scores2, labels)) / scores1.size(0) 


class criterion_backbone(nn.Module):
    # 286.FuM-ADS: Fomular 5, Fomular 2,4
    # train_ADA,py: line 68-78 (batchsize=128)
    def __init__(self, tao=0.1, alpha=128):
        super(criterion_backbone, self).__init__()
        self.tao = tao
        self.alpha = alpha # train_ADS.py, line 78, alpha=128=batchsize, in 286.FuM-ADS.pdf, alpha should be batchsize, but why batchsize is involved in loss calculation in one graph?  
        self.metric = metric_entropy()

    def forward(self, scores1, scores2):
        loss = torch.mean(torch.abs(scores1 - scores2), dim=1) # fomular 2
        loss_weight = (1- torch.sigmoid(self.metric(scores1, scores2) - self.tao)) / self.alpha # fomular 4
        return torch.mean(loss_weight.detach() * loss) # fomular 5


class criterion_classifier(nn.Module):
    # 286.FuM-ADS: Fomular 8, Fomular 6,7
    # train_ADA,py: line 127-138 (batchsize=128)
    def __init__(self, tao=0.1, alpha=128):
        super(criterion_classifier, self).__init__()
        self.tao = tao # tao=0.1
        self.alpha = alpha # train_ADS.py, line 137, alpha=128=batchsize, in 286.FuM-ADS.pdf, alpha should be batchsize, but why batchsize is involved in loss calculation in one graph?  
        self.metric = metric_entropy()

    def forward(self, scores1, scores2):
        loss = 1 - torch.mean(torch.abs(scores1 - scores2), dim=1) # fomular 6
        loss_weight = torch.sigmoid(self.metric(scores1, scores2) - self.tao) / self.alpha # fomular 7
        return torch.mean(loss_weight.detach() * loss) # fomular 8


class ADSNet_backbone(ResNet_Cifar):
    # models/resnet.py: ResNet line 115-120
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10):
        super(ADSNet_backbone, self).__init__(block=block, num_blocks=num_blocks, num_classes=num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out


class ADSNet_classifier(nn.Module):
    # models/resnet.py: ResNet line 122-130
    def __init__(self, alpha=0.05, num_proto=3, num_classes=10, block_expansion=1):
        super(ADSNet_classifier, self).__init__()
        self.alpha = alpha
        self.num_proto = num_proto
        self.num_classes = num_classes
        self.channels = 512*block_expansion
        self.conv  = nn.Conv2d(self.channels, self.channels, kernel_size=1, bias=False)
        self.protos = nn.Parameter(torch.randn(self.num_proto * self.num_classes, self.channels), requires_grad=True)

    @staticmethod
    def cosine_distance(feat1, feat2):
        return torch.matmul(F.normalize(feat1), F.normalize(feat2).t())

    def forward(self, x):
        # input x: feature (B,C=512,1,1)
        # before F.avg_pool2d(out, 4) in ResNet in "https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py", feature is (B,C,4,4)
        x = torch.sigmoid(self.conv(x).view(x.size(0), -1))
        cls_score, _ = self.cosine_distance(x, self.protos).view(-1, self.num_proto, self.num_classes).max(dim=1)
        return torch.sigmoid(cls_score / self.alpha)


class ADSNet(nn.Module):
    # models/resnet.py: ResNet
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10):
        # ResNet18: BasicBlock, [2, 2, 2, 2]
        # ResNet34: BasicBlock, [3, 4, 6, 3]
        # ResNet50: Bottleneck, [3, 4, 6, 3]
        super(ADSNet, self).__init__()
        self.backbone = ADSNet_backbone(block=block, num_blocks=num_blocks, num_classes=num_classes)
        self.classifier1 = ADSNet_classifier(num_classes=num_classes, block_expansion=block.expansion)
        self.classifier2 = ADSNet_classifier(num_classes=num_classes, block_expansion=block.expansion)

    def freeze_backbone(self):
        for _, (k, v) in enumerate(self.named_parameters()):
            if k.startswith('backbone'):
                v.requires_grad = False
            else:
                v.requires_grad = True
        params = filter(lambda p: p.requires_grad, self.parameters())   
        return params    

    def freeze_classifier(self):
        ''' ipynb
        ## when classifier is frozen, backbone can still have gradients
        import torch, main_ADS
        net = main_ADS.ADSNet().cuda()
        net.freeze_classifier()
        score1, score2 = net(torch.randn(1,3,32,32).cuda())
        (torch.sum(score1) + torch.sum(score2)).backward()
        for _, (k,v) in enumerate(net.named_parameters()):
            print(k, v.grad)
        '''
        for _, (k, v) in enumerate(self.named_parameters()):
            if k.startswith('classifier'):
                v.requires_grad = False
            else:
                v.requires_grad = True
        params = filter(lambda p: p.requires_grad, self.parameters())   
        return params

    def unfreeze_all(self):
        for _, (_, v) in enumerate(self.named_parameters()):
            v.requires_grad = True
        return self.parameters()

    def forward(self, x):
        out = self.backbone(x)
        score1 = self.classifier1(out)
        score2 = self.classifier2(out)

        return score1, score2


def get_one_hot_label(labels=None, num_classes=10):
    # from labels: [0, 1, 2] to labels:[[1,0,0,0],[0,1,0,0],[0,0,1,0]] (Ncls=4)
    return torch.zeros(labels.shape[0], num_classes, device=labels.device).scatter_(1, labels.view(-1, 1), 1)


def train_init(model=ADSNet().cuda(), _criterion_init=criterion_init(), labeled_loader=None, optimizer=None):
    # train backbone and classifier
    model.train()
    model.unfreeze_all()

    loss = 0
    for data in labeled_loader:
        inputs = data[0].cuda()
        labels = get_one_hot_label(data[1]).cuda()

        optimizer.zero_grad()
        scores1, scores2 = model(inputs)
        loss = _criterion_init(scores1, scores2, labels)
        loss.backward()
        optimizer.step()

    return loss


def train_backbone(model=ADSNet().cuda(), _criterion_backbone=criterion_backbone(), unlabeled_loader=None, optimizer=None):
    model.train()
    params = model.freeze_classifier()

    loss = 0
    for data in unlabeled_loader:
    
        params = model.freeze_classifier()
        optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160])
    
        inputs = data[0].cuda()
        scores1, scores2 = model(inputs)
        loss = _criterion_backbone(scores1, scores2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

    return loss


def train_classifier(model=ADSNet().cuda(), _criterion_init=criterion_init(), _criterion_classifier=criterion_classifier(), labeled_loader=None, unlabeled_loader=None, optimizer=None):
    model.train()
    params = model.freeze_backbone()

    # train_ADS.py line 64: define optim after model.forward? is that right?
    # why define optim and scheduler in one epoch?
    # train_ADS.py line 154: there's another scheduler? 
    #..... if model.parameters() is right?
    # if optimizer is None:
    # optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005)
    # scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160])
    
    loss = 0
    inputlist = list(enumerate(labeled_loader))
    inputlist_len = len(inputlist)
    for i, (uinputs, _) in enumerate(unlabeled_loader):
        (_, (inputs, labels)) = inputlist[i % inputlist_len]
        inputs = inputs.cuda()
        labels = get_one_hot_label(labels).cuda()

        params = model.freeze_backbone()
        optimizer = optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=0.0005)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[160])

        uinputs = uinputs.cuda()
        scores1, scores2 = model(inputs)
        uscores1, uscores2 = model(uinputs)
        # why train these two in the sametime ?
        loss = _criterion_init(scores1, scores2, labels) + _criterion_classifier(uscores1, uscores2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

    return loss


def train_cycle(cfg, model, train_loader, unlabeled_loader):
    pass


def test_cycle(model, test_loader):
    model.eval()

    total = 0
    correct1 = 0
    correct2 = 0
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs = inputs.cuda()
            labels = labels.cuda()

            scores1, scores2 = model(inputs)
            _, preds1 = torch.max(scores1.data, 1)
            _, preds2 = torch.max(scores2.data, 1)
            total += labels.size(0)
            correct1 += (preds1 == labels).sum().item()
            correct2 += (preds2 == labels).sum().item()
            acc1 = 100 * correct1 / total
            acc2 = 100 * correct2 / total
            acc = (100 * correct1 / total + 100 * correct2 / total) * 0.5

    return acc1, acc2, acc


def get_uncertainty(model, unlabeled_loader):
    model.eval()
    uncertainty = torch.tensor([]).cuda()
    metric = metric_entropy()
    with torch.no_grad():
        for (uinputs, _) in unlabeled_loader:
            uinputs = uinputs.cuda()
            scores1, scores2 = model(uinputs)
            uncertainty = torch.cat((uncertainty, metric(scores1, scores2)), dim=0)
    return uncertainty.cpu()


def train_ADS(dataset='cifar10'):
    cfg = config.CONFIG()
    checkpoint_dir =  os.path.realpath(os.path.join('./output', time.strftime('%m%d%H%M%S', time.localtime())))
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    log = printlog(checkpoint_dir + '/log.log')

    for k, v in cfg.__dict__.items():
        try:
           for k1, v1 in v.__dict__.items(): 
               log.log(f'{k1}:{v1}')
        except:
            log.log(f'{k}:{v}')
    log.log(f'checkpoint:{checkpoint_dir}')
        
    train_dataset, test_dataset, unlabeled_dataset = get_dataset(dataset, cfg.DATASET.ROOT[dataset])
    test_loader = DataLoader(test_dataset, batch_size=cfg.TRAIN.BATCH)

    Performance = np.zeros((3, 10))
    log.log('Train Start.')
    for trial in range(cfg.ACTIVE_LEARNING.TRIALS):

        torch.backends.cudnn.benchmark = True
        model = ADSNet(block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10).cuda()

        indices = list(range(cfg.DATASET.NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:cfg.ACTIVE_LEARNING.ADDENDUM]
        unlabeled_set = indices[cfg.ACTIVE_LEARNING.ADDENDUM:]
        train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH, sampler=SubsetRandomSampler(labeled_set), pin_memory=True)

        for cycle in range(cfg.ACTIVE_LEARNING.CYCLES):
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:cfg.ACTIVE_LEARNING.SUBSET]
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=cfg.TRAIN.BATCH, sampler=SubsetSequentialSampler(subset), pin_memory=True)

            optim_init = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WDECAY)
            optim_backbone = optim.SGD(model.freeze_classifier(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WDECAY)
            optim_classifier = optim.SGD(model.freeze_backbone(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=cfg.TRAIN.WDECAY)
            scheduler = lr_scheduler.MultiStepLR(optim_init, milestones=cfg.TRAIN.MILESTONES)

            _criterion_init=criterion_init()
            _criterion_backbone=criterion_backbone()
            _criterion_classifier=criterion_classifier()

            for epoch in range(cfg.TRAIN.EPOCH):
                ## why inside opoch?
                # optimizer_backbone = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=0.0005)
                # optimizer_classifier = optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=0.0005)

                if epoch == 0:
                    train_init(model=model, _criterion_init=_criterion_init, labeled_loader=train_loader, optimizer=optim_init)
                loss_b = train_backbone(model=model, _criterion_backbone=_criterion_backbone, unlabeled_loader=unlabeled_loader, optimizer=optim_backbone)
                loss_c = train_classifier(model=model, _criterion_init=_criterion_init, _criterion_classifier=_criterion_classifier, labeled_loader=train_loader, unlabeled_loader=unlabeled_loader, optimizer=optim_classifier)
                loss_i = train_init(model=model, _criterion_init=_criterion_init, labeled_loader=train_loader, optimizer=optim_init)
            
                scheduler.step()

                if epoch == 0 or (epoch + 1) % 10 == 0:
                    log.log(f'Trial [{trial + 1}/{cfg.ACTIVE_LEARNING.TRIALS}]',
                        f'Cycle [{cycle + 1}/{cfg.ACTIVE_LEARNING.CYCLES}]',
                        f'Epoch [{epoch + 1}/{cfg.TRAIN.EPOCH}]',
                        f'Labeled Set Size [{len(labeled_set)}]',
                        f'Loss [backbone: {loss_b:.3g}, Loss_classifier: {loss_c:.3g}, Loss_init: {loss_i:.3g}]'
                    )

            acc1, acc2, acc = test_cycle(model, test_loader)
            Performance[trial, cycle] = acc

            log.log(f'Trial [{trial + 1}/{cfg.ACTIVE_LEARNING.TRIALS}]',
                        f'Cycle [{cycle + 1}/{cfg.ACTIVE_LEARNING.CYCLES}]',
                        f'Labeled Set Size [{len(labeled_set)}]',
                        f'Test [acc1: {acc1:.3g}, acc2: {acc2:.3g}, acc: {acc:.3g}]'
            )

            arg = np.argsort(get_uncertainty(model, unlabeled_loader))
            labeled_set += list(torch.tensor(subset)[arg][-cfg.ACTIVE_LEARNING.ADDENDUM:].numpy())
            unlabeled_set = list(torch.tensor(subset)[arg][:-cfg.ACTIVE_LEARNING.ADDENDUM].numpy()) + unlabeled_set[cfg.ACTIVE_LEARNING.SUBSET:]
            train_loader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH, sampler=SubsetRandomSampler(labeled_set), pin_memory=True)

            torch.save(model.state_dict(), f'{checkpoint_dir}/ADSNet_trial_{trial}_cycle_{cycle}_epoch_{cfg.TRAIN.EPOCH}.pth')

    log.log('Performance Summary: ', withtime=False)
    for trial in range(cfg.ACTIVE_LEARNING.TRIALS):
        log.log(f'Trail {trial + 1}: {Performance[trial]}', withtime=False)
    
    log.close()


if __name__ == '__main__':
    train_ADS()
