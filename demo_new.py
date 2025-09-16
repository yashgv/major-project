import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
from scipy.io import savemat
from torch import optim
from enhanced_model import EnhancedS2VNet
from enhanced_modules import EnhancedLossFunction, HSIAugmentation
from utils import AvgrageMeter, accuracy, output_metric, NonZeroClipper, print_args
from dataset import prepare_dataset
import numpy as np
import time
import os

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--fix_random', action='store_true', default=True, help='fix randomness')
parser.add_argument('--gpu_id', default='1', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--dataset', choices=['Indian', 'Berlin', 'Augsburg'], default='Indian', help='dataset to use')
parser.add_argument('--flag_test', choices=['test', 'train'], default='train', help='testing mark')
parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=7, help='number of patches')
parser.add_argument('--epoches', type=int, default=500, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay')  # Added L2 regularization
parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing factor')
args = parser.parse_args()

def train_epoch(model, train_loader, criterion, optimizer, augmentation):
    model.train()
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])

    for batch_idx, (batch_data, batch_target) in enumerate(train_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()

        # Ensure data is in the correct format and type
        if batch_data.dtype != torch.float32:
            batch_data = batch_data.to(torch.float32)
        if batch_target.dtype != torch.long:
            batch_target = batch_target.to(torch.long)
            
        # Apply augmentations
        batch_data, batch_target = augmentation(batch_data, batch_target)

        optimizer.zero_grad()
        output = model(batch_data)

        # Calculate loss
        loss = criterion(output, batch_target)
        
        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(output, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.item(), n)
        top1.update(prec1[0].item(), n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre

def validate(model, val_loader, criterion):
    model.eval()
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])

    with torch.no_grad():
        for batch_idx, (batch_data, batch_target) in enumerate(val_loader):
            batch_data = batch_data.cuda()
            batch_target = batch_target.cuda()
            
            # Ensure data is in the correct format and type
            if batch_data.dtype != torch.float32:
                batch_data = batch_data.to(torch.float32)
            if batch_target.dtype != torch.long:
                batch_target = batch_target.to(torch.long)

            output = model(batch_data)
            loss = criterion(output, batch_target)

            prec1, t, p = accuracy(output, batch_target, topk=(1,))
            n = batch_data.shape[0]
            objs.update(loss.item(), n)
            top1.update(prec1[0].item(), n)
            tar = np.append(tar, t.data.cpu().numpy())
            pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, objs.avg, tar, pre

def main():
    if not torch.cuda.is_available():
        print('GPU is not available')
        return
    
    if args.fix_random:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print_args(vars(args))

    # Prepare dataset
    train_loader, test_loader, true_loader, band, height, width, num_classes, label, total_pos_true = prepare_dataset(args)
    
    # Initialize model
    model = EnhancedS2VNet(band=band, num_classes=num_classes, patch_size=args.patches)
    model = model.cuda()
    
    # Initialize advanced loss function
    criterion = EnhancedLossFunction(
        n_classes=num_classes, 
        alpha=0.25,
        gamma=2.0,
        label_smoothing=args.label_smoothing
    ).cuda()
    
    # Initialize optimizer with warmup and cosine annealing
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=50,  # Initial restart interval
        T_mult=2  # Multiply T_0 by this factor after each restart
    )
    
    # Initialize data augmentation
    augmentation = HSIAugmentation(p=0.5)

    best_acc = 0
    for epoch in range(args.epoches):
        # Training
        train_acc, train_obj, tar_t, pre_t = train_epoch(
            model, train_loader, criterion, optimizer, augmentation
        )
        scheduler.step()
        
        # Validation
        if (epoch + 1) % args.test_freq == 0:
            val_acc, val_obj, tar_v, pre_v = validate(model, test_loader, criterion)
            
            if val_acc > best_acc:
                best_acc = val_acc
                state = {
                    'epoch': epoch,
                    'args': args,
                    'best_acc': best_acc,
                    'model': model.state_dict()
                }
                torch.save(
                    state,
                    f'checkpoints/{args.dataset}_enhanced_s2vnet_p{args.patches}_{val_acc:.2f}_epoch{epoch+1}.pth'
                )
            
            print(f'Epoch {epoch+1:3d}/{args.epoches:3d}:')
            print(f'Train - Loss: {train_obj:.4f}, Acc: {train_acc:.2f}%')
            print(f'Val   - Loss: {val_obj:.4f}, Acc: {val_acc:.2f}%')
            print(f'Best accuracy: {best_acc:.2f}%')

if __name__ == '__main__':
    main()
