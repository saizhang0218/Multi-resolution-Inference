import argparse
import os
import random
import shutil
import time
import warnings
from collections import OrderedDict 

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import models
from models.quant_layer import *
from tensorboardX import SummaryWriter
import sys
import gc
import util
import dataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('save_path', help='path to save model')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18', choices=model_names)
parser.add_argument('-j','--workers', default=4, type=int, metavar='N',help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=60, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=768, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--bit', default=5, type=int, help='the bit-width of the multi-resolution model')
parser.add_argument('--data', metavar='DATA_PATH', default='./data/', help='path to imagenet data (default: ./data/)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('-p', '--print-freq', default=30, type=int, metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
parser.add_argument('-lt', '--loss_type', default='joint', help='two options: distillation, joint')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

'''
The list of quantization bitwidth.
For example, [a,b] represents jointly training of two sub-models.
The first sub-model adopts a bits activations and weights.
The second sub-model adopts b bits activations and weights.
'''

quant_list = [3,4,5]

def main():
    args = parser.parse_args()
    if not os.path.exists('result'):
        os.makedirs('result')
    fdir = 'result/'+args.save_path + str('.pth')
    if not os.path.exists(fdir):
        os.makedirs(fdir)
        
    fdir_acc = 'accuracy/' + args.save_path
    if not os.path.exists(fdir_acc):
        os.makedirs(fdir_acc)
        
        
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, fdir)


def main_worker(gpu, ngpus_per_node, args, fdir):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch](pretrained=True, weight_bit=args.bit, data_bit=args.bit)
    
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    #############################
    # Load the pretrained model
    #############################
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading pre-trained model from {}".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            model.load_state_dict(checkpoint['state_dict'])
        else:
            print('no pre-trained model found')
            exit()

    # define loss function and optimizer
    if args.loss_type == 'joint':
        criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    elif args.loss_type == 'distill':
        criterion = util.loss_teacher_student
    criterion_test = nn.CrossEntropyLoss().cuda(args.gpu)
    model_params = []
    for name, params in model.module.named_parameters():
        if 'act_alpha' in name:
            model_params += [{'params': [params], 'lr': 3e-2, 'weight_decay': 2e-5}]
        elif 'wgt_alpha' in name:
            model_params += [{'params': [params], 'lr': 1e-2, 'weight_decay': 1e-4}]
        else:
            model_params += [{'params': [params]}]
    optimizer = torch.optim.SGD(model_params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1e+6))

    cudnn.benchmark = True


    #############################
    # Data loader of ImageNet
    #############################
        print('==> Using Pytorch Dataset')
    input_size = 224  # image resolution for resnets
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    torchvision.set_image_backend('accimage')
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    
    #############################
    # Writing the summary
    #############################
    writer = SummaryWriter(comment='res18_4bit')
    
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)
        # evaluate on validation set
        top1_m_avg_list = validate(val_loader, model, criterion_test, args)
        
        for j in range(len(top1_m_avg_list)):
            f = open('./accuracy/' + args.save_path + '/' + 'bitwidth' + str(j) + "_val_acc.txt", "a")
            f.write(str(top1_m_avg_list[j].cpu().numpy()) + ',')
        f.close()

        
def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    global quant_list
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_m1 = AverageMeter('Acc@1', ':6.2f')
    top5_m1 = AverageMeter('Acc@5', ':6.2f')
    top1_m2 = AverageMeter('Acc@1', ':6.2f')
    top5_m2 = AverageMeter('Acc@5', ':6.2f')
    
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1_m1, top5_m1, top1_m2, top5_m2],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    
    quant_list_temp = quant_list
    counter = 0
    for i, (images, target) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        if (counter == len(quant_list_temp)):
            random.shuffle(quant_list_temp)
            counter = 0
        current_quant1 = quant_list[-1]
        current_quant2 = quant_list_temp[counter]
        counter = counter + 1    
        
        
        output1 = model(images, current_quant1)
        output2 = model(images, current_quant2)
        
        if (args.loss_type == 'joint'): 
            loss = criterion(output1, target) + criterion(output2, target)
        elif (args.loss_type == 'distill'):
            loss = criterion(output2, target, output1)
            
        # measure accuracy and record loss
        acc1_m1, acc5_m1 = accuracy(output1, target, topk=(1, 5))
        acc1_m2, acc5_m2 = accuracy(output2, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1_m1.update(acc1_m1[0], images.size(0))
        top5_m1.update(acc5_m1[0], images.size(0))
        top1_m2.update(acc1_m2[0], images.size(0))
        top5_m2.update(acc5_m2[0], images.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        gc.collect()
    writer.add_scalar('train_acc', top1_m1.avg, epoch)


def validate(val_loader, model, criterion, args):
    global quant_list
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1_m_list = []
    top5_m_list = []
    for _ in quant_list:
        top1_m_list.append(AverageMeter('Acc@1', ':6.2f'))
        top5_m_list.append(AverageMeter('Acc@5', ':6.2f'))
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1_m_list[0], top5_m_list[0]],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            loss = 0.0
            for j in range(len(quant_list)):
                # compute output
                output = model(images, quant_list[j])
                loss += criterion(output, target)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                top1_m_list[j].update(acc1[0], images.size(0))
                top5_m_list[j].update(acc5[0], images.size(0))
                
            losses.update(loss.item(), images.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        top1_m_avg_list = []
        for j in range(len(quant_list)):
            top1_m_avg_list.append(top1_m_list[j].avg)
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Bitwidth {bitwidth:.1f}'.format(top1=top1_m_list[j], top5=top5_m_list[j], bitwidth=quant_list[j]))

    return top1_m_avg_list


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    adjust_list = [15,30,40,50]
    if epoch in adjust_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
