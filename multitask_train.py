from __future__ import print_function

import argparse
import os
import shutil
import time
import random

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

from models import resynet
from load_patches_data import PatchesDataset, PatchesDatasetSubtype
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p, savefig
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch MixMatch Training')
# Optimization options
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch-size', default=18, type=int, metavar='N',
                    help='train batchsize')
parser.add_argument('--num_workers', default=4, type=int,
                    metavar='N', help='number of workers')
parser.add_argument('--lr', '--learning-rate', default=0.002, type=float,
                    metavar='LR', help='initial learning rate')
# Checkpoints
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--transfer', default=False, type=bool)
# Miscs
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
#Device options
parser.add_argument('--gpu', default='0,1', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
#Method options
#parser.add_argument('--n-labeled', type=int, default=250,
#                        help='Number of labeled data')
parser.add_argument('--val-iteration', type=int, default=1024,
                        help='Number of labeled data')
parser.add_argument('--out', default='/home5/hby/subtype_newdata/0.3_3/yfy/our_retune',
                        help='Directory to output the result')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=40, type=float)
parser.add_argument('--mu-u', default=1, type=float)
parser.add_argument('--s-start-epochs', default=5, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)


args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy
num_classes_1 = 2 # cancer vs normal
num_classes_2 = 3 # ccrcc vs prcc vs chrcc

def main():
    global best_acc

    if not os.path.isdir(args.out):
        mkdir_p(args.out)

    # Data
    print(f'==> Preparing dataset')
    labeled_data_files = "./data/RCC/labeled_2000_train.txt"
    unlabeled_files = "./data/RCC/unlabeled_2000_train.txt"
    test_files =  "./data/RCC/all_2000_test.txt"
    print(labeled_data_files)
    print(unlabeled_files)
    print(test_files)
    labeled_trainloader, unlabeled_trainloader, val_loader = \
        build_dataset(labeled_data_files, unlabeled_files, test_files)
    

    # Model
    print("==> creating resynet 34")

    def create_model(ema=False):
        model = resynet.resnet34(pretrained=True, num_classes_1=num_classes_1, num_classes_2=num_classes_2)
        model = model.cuda()
        model = torch.nn.DataParallel(model)
        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model()
    ema_model = create_model(ema=True)

    cudnn.benchmark = True
    print('    Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))

    train_criterion_D = SemiLoss()
    train_criterion_S = SubtypeLoss()
    criterion = nn.CrossEntropyLoss()
    weight = torch.Tensor([1,1,1])
    criterion_subtype = nn.CrossEntropyLoss(weight=weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    ema_optimizer= WeightEMA(model, ema_model, alpha=args.ema_decay)
    start_epoch = 0

    # Resume
    title = 'framework3-region-detection-subtype'
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        ema_model.load_state_dict(checkpoint['ema_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title, resume=True)
        print(start_epoch,best_acc)

    else:
        logger = Logger(os.path.join(args.out, 'log.txt'), title=title)
        logger.set_names(['Train Loss', 'Train Loss X', 'Train Loss U',  'Valid Loss', 'Valid Acc.'])

    writer = SummaryWriter(args.out)
    step = 0
#    test_accs = []
    # Train and val
    for epoch in range(start_epoch, args.epochs):

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, train_criterion_D, train_criterion_S, epoch, use_cuda)
        _, train_acc = validate(labeled_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats')
        val_loss, val_acc = validate(val_loader, ema_model, criterion, epoch, use_cuda, mode='Valid Stats')
#        test_loss, test_acc = validate(test_loader, ema_model, criterion, epoch, use_cuda, mode='Test Stats ')

        step = args.val_iteration * (epoch + 1)

        writer.add_scalar('losses/train_loss', train_loss, step)
        writer.add_scalar('losses/valid_loss', val_loss, step)
#        writer.add_scalar('losses/test_loss', test_loss, step)

        writer.add_scalar('accuracy/train_acc', train_acc, step)
        writer.add_scalar('accuracy/val_acc', val_acc, step)
#        writer.add_scalar('accuracy/test_acc', test_acc, step)

        # append logger file
#        logger.append([train_loss, train_loss_x, train_loss_u, val_loss, val_acc, test_loss, test_acc])
        logger.append([train_loss, train_loss_x, train_loss_u, val_loss, val_acc])
        
        # save model
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'acc': val_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            }, is_best)
#        test_accs.append(test_acc)
    logger.close()
    writer.close()

    print('Best acc:')
    print(best_acc)

#    print('Mean acc:')
#    print(np.mean(test_accs[-20:]))

def build_dataset(meta_data_files, train_files, test_files):
    class TransformTwice:
        def __init__(self, transform):
            self.transform = transform
    
        def __call__(self, inp):
            out1 = self.transform(inp)
            out2 = self.transform(inp)
            return out1, out2
        
    normMean = [0.744, 0.544, 0.670]
    normStd = [0.183, 0.245, 0.190]
    normTransform = transforms.Normalize(normMean, normStd)
    
    if True:
        train_transform = transforms.Compose([
            transforms.Resize(512),
            transforms.RandomCrop(512, padding=64, padding_mode='reflect'),
            transforms.RandomHorizontalFlip(),#图像一半的概率翻转，一半的概率不翻转
            transforms.RandomRotation((-90,90)), #随机旋转
            transforms.ToTensor(),
            normTransform,
        ])
    else:
        train_transform = transforms.Compose([
            #transforms.Resize(224),
            transforms.ToTensor(),
            normTransform,
        ])
    test_transform = transforms.Compose([
        transforms.Resize(512),
        transforms.ToTensor(),
        normTransform
    ])
    meta_data = PatchesDatasetSubtype(meta_data_files, transform=train_transform)
    meta_loader = torch.utils.data.DataLoader(
            meta_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    train_data = PatchesDatasetSubtype(train_files, transform=TransformTwice(train_transform))
    train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_data = PatchesDatasetSubtype(test_files, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)
    return meta_loader, train_loader, test_loader

def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion_D, criterion_S, epoch, use_cuda):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_Dx = AverageMeter()
    losses_Du = AverageMeter()
    losses_Sx = AverageMeter()
    losses_Su = AverageMeter()
    ws = AverageMeter()
    us = AverageMeter()
    end = time.time()

    bar = Bar('Training', max=args.val_iteration)
    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    
    model.train()
    for batch_idx in range(args.val_iteration):
        try:
            inputs_x, targets_x, subtypes_x, _ = labeled_train_iter.next()
        except:
            labeled_train_iter = iter(labeled_trainloader)
            inputs_x, targets_x, subtypes_x, _ = labeled_train_iter.next()

        try:
            (inputs_u, inputs_u2), _, subtypes_u, _ = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (inputs_u, inputs_u2), _, subtypes_u, _ = unlabeled_train_iter.next()

        # measure data loading time
        data_time.update(time.time() - end)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = torch.zeros(batch_size, num_classes_1).scatter_(1, targets_x.long().view(-1,1), 1)
        subtypes_x = torch.zeros(batch_size, num_classes_2).scatter_(1, subtypes_x.long().view(-1,1), 1)
        subtypes_u = torch.zeros(batch_size, num_classes_2).scatter_(1, subtypes_u.long().view(-1,1), 1)

        if use_cuda:
            inputs_x, targets_x, subtypes_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True), subtypes_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()
            inputs_u2 = inputs_u2.cuda()
            subtypes_u = subtypes_u.cuda(non_blocking=True)


        with torch.no_grad():
            # compute guessed labels of unlabel samples
            outputs_u, _ = model(inputs_u)
            outputs_u2, _ = model(inputs_u2)
            p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
            pt = p**(1/args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)
        all_subtypes = torch.cat([subtypes_x, subtypes_u, subtypes_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1-l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]
        subtype_a, subtype_b = all_subtypes, all_subtypes[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b
        mixed_subtype =  l * subtype_a + (1 - l) * subtype_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        outputs_0, outputs_1 = model(mixed_input[0])
        logits_D = [outputs_0]
        logits_S = [outputs_1]
        for input in mixed_input[1:]:
            outputs_0, outputs_1 = model(input)
            logits_D.append(outputs_0)
            logits_S.append(outputs_1)

        # put interleaved samples back
        logits_D = interleave(logits_D, batch_size)
        logits_Dx = logits_D[0]
        logits_Du = torch.cat(logits_D[1:], dim=0)
        logits_S = interleave(logits_S, batch_size)
        logits_Sx = logits_S[0]
        logits_Su = torch.cat(logits_S[1:], dim=0)
        
        LDx, LDu, w = criterion_D(logits_Dx, mixed_target[:batch_size], logits_Du, mixed_target[batch_size:], epoch+batch_idx/args.val_iteration)
        LSx, LSu, u = criterion_S(logits_Sx, mixed_target[:batch_size], mixed_subtype[:batch_size], logits_Su, mixed_target[batch_size:], mixed_subtype[batch_size:], epoch+batch_idx/args.val_iteration)
        
        if args.transfer:
            lossD = LDx + args.lambda_u * LDu
            lossS = LSx + args.mu_u * LSu
        else:
            lossD = LDx + w * LDu
            lossS = LSx + u * LSu
        loss = lossD + lossS

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_Dx.update(LDx.item(), inputs_x.size(0))
        losses_Du.update(LDu.item(), inputs_x.size(0))
        losses_Sx.update(LSx.item(), inputs_x.size(0))
        losses_Su.update(LSu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))
        us.update(u, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | L: {loss:.3f} | LDx: {loss_Dx:.3f} | LDu: {loss_Du:.3f} | W: {w:.3f} | LSx: {loss_Sx:.3f} | LSu: {loss_Su:.3f} | U: {u:.3f}'.format(
                    batch=batch_idx + 1,
                    size=args.val_iteration,
                    data=data_time.avg,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    loss_Dx=losses_Dx.avg,
                    loss_Du=losses_Du.avg,
                    loss_Sx=losses_Sx.avg,
                    loss_Su=losses_Su.avg,
                    w=ws.avg,
                    u=us.avg,
                    )
        bar.next()
    bar.finish()

    return (losses.avg, losses_Dx.avg, losses_Du.avg,)

def validate(valloader, model, criterion, epoch, use_cuda, mode):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    lossesS = AverageMeter()
    lossesD = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
#    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar(f'{mode}', max=len(valloader))
    with torch.no_grad():
        for batch_idx, (inputs, targets, subtypes, _) in enumerate(valloader):
            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets, subtypes = inputs.cuda(), targets.cuda(non_blocking=True).long(), subtypes.cuda(non_blocking=True).long()
            
            # compute output
            outputs_0, outputs_1 = model(inputs)
            lossD = criterion(outputs_0, targets)
            
            indexes = [i for i in range(len(targets)) if targets[i]==0]
            if len(indexes) == 0:
                
                lossS = torch.zeros(1)
            else:
                subtypes = subtypes[indexes]
                outputs_1 = outputs_1[indexes]
                lossS = criterion(outputs_1, subtypes)

            # measure accuracy and record loss
            prec1 = accuracy(outputs_0, targets, topk=(1,))
            prec2 = accuracy(outputs_1, subtypes, topk=(1,))
            lossesD.update(lossD.item(), inputs.size(0))
            lossesS.update(lossS.item(), inputs.size(0))
            top1.update(prec1[0].item(), inputs.size(0))
            top2.update(prec2[0].item(), inputs.size(0))
#            top5.update(prec5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | LD: {lossD:.4f} | top1: {top1: .4f} | LS: {lossS:.4f} | top2: {top2: .4f}'.format(
                        batch=batch_idx + 1,
                        size=len(valloader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        lossD=lossesD.avg,
                        top1=top1.avg,
                        lossS=lossesS.avg,
                        top2=top2.avg,
#                        top5=top5.avg,
                        )
            bar.next()
        bar.finish()
    return (lossesD.avg, top1.avg)

def save_checkpoint(state, is_best, checkpoint=args.out, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)

class SubtypeLoss(object):
    def __call__(self, outputs_x, targets_x, subtypes_x, outputs_u, targets_u, subtypes_u, epoch):
        
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * subtypes_x, dim=1) * targets_x[:,0])
        Lu = -torch.mean(torch.sum(F.log_softmax(outputs_u, dim=1) * subtypes_u, dim=1) * targets_u[:,0])

        if epoch < args.s_start_epochs:
            epoch = 0
            
        return Lx, Lu, args.mu_u * linear_rampup(epoch)

class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param = ema_param

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.float()
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param = param.float()
            param.mul_(1 - self.wd)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

if __name__ == '__main__':
    main()
