import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim
import numpy as np
import os
import math
import argparse
from model import S2VNetU
from utils import AvgrageMeter, accuracy, output_metric
from dataset import prepare_dataset

def parse_args():
    parser = argparse.ArgumentParser("S2VNet-U Training")
    parser.add_argument('--fix_random', action='store_true', default=True, help='fix randomness')
    parser.add_argument('--gpu_id', default='0', help='gpu id')
    parser.add_argument('--seed', type=int, default=0, help='number of seed')
    parser.add_argument('--dataset', choices=['Indian', 'Berlin', 'Augsburg'], default='Indian', help='dataset to use')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--epoches', type=int, default=300, help='total number of epoch')
    parser.add_argument('--test_freq', type=int, default=1, help='test frequency')
    parser.add_argument('--dropout_rate', type=float, default=0.3, help='dropout rate for all layers')
    parser.add_argument('--l2_reg', type=float, default=1e-5, help='L2 regularization strength')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label smoothing factor')
    parser.add_argument('--patches', type=int, default=1, help='patches size')
    parser.add_argument('--dropout_p', type=float, default=0.2, help='dropout probability')
    parser.add_argument('--mc_samples_train', type=int, default=8, help='number of MC samples during training')
    parser.add_argument('--mc_samples_eval', type=int, default=32, help='number of MC samples during evaluation')
    parser.add_argument('--lambda_cls', type=float, default=1.0, help='classification loss weight')
    parser.add_argument('--lambda_cons', type=float, default=0.1, help='consistency loss weight')
    parser.add_argument('--fast_debug', action='store_true', help='use small subset and fewer epochs for debugging')
    return parser.parse_args()

def dirichlet_kl_loss(alpha, alpha0=0.01):
    """Compute KL divergence between predicted Dirichlet and symmetric prior"""
    num_classes = alpha.shape[1]
    alpha0 = torch.full_like(alpha, alpha0)
    
    # Compute KL divergence terms
    return torch.sum(
        torch.lgamma(alpha.sum(1)) - torch.lgamma(alpha0.sum(1)) -
        torch.sum(torch.lgamma(alpha), dim=1) + torch.sum(torch.lgamma(alpha0), dim=1) +
        torch.sum((alpha - alpha0) * (torch.digamma(alpha) - 
                 torch.digamma(alpha.sum(1, keepdim=True))), dim=1)
    )

def compute_consistency_loss(abundance_samples, temperature=1.0):
    """Compute consistency loss across MC samples"""
    mean_abundance = abundance_samples.mean(dim=0, keepdim=True)
    mean_abundance = mean_abundance.expand_as(abundance_samples)
    return F.mse_loss(abundance_samples, mean_abundance) * temperature

def weighted_ce_loss(logits, labels, confidence, label_smoothing=0.0):
    """Compute confidence-weighted cross entropy loss with label smoothing"""
    # Get batch size and number of classes
    batch_size = logits.size(0)
    num_classes = logits.size(1)
    
    # Average confidence per sample if needed
    if confidence.numel() != batch_size:
        confidence = confidence.view(batch_size, -1).mean(dim=1)
    
    # Compute weights
    weights = torch.clamp(1.0 + (confidence - 0.5), min=0.1, max=1.5)
    
    # Create smooth labels
    with torch.no_grad():
        smooth_labels = torch.zeros_like(logits)
        smooth_labels.fill_(label_smoothing / (num_classes - 1))
        smooth_labels.scatter_(1, labels.unsqueeze(1), 1 - label_smoothing)
    
    # Compute KL div loss (equivalent to CE with label smoothing)
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(smooth_labels * log_probs).sum(dim=1)
    
    # Apply weights
    weighted_loss = weights * loss
    return weighted_loss.mean()

def train_epoch(model, train_loader, criterion, optimizer, scheduler, args):
    model.train()
    losses = AvgrageMeter()
    top1 = AvgrageMeter()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()
        
        # Forward pass
        outputs = model(data, is_training=True)
        
        # Compute losses
        reconstruction_loss = F.mse_loss(outputs['reconstruction'], data)
        dirichlet_loss = dirichlet_kl_loss(outputs['abundances']['alpha'])
        consistency_loss = compute_consistency_loss(
            outputs['abundances']['samples'],
            temperature=1.0
        )
        
        # Cross entropy with label smoothing
        ce_loss = weighted_ce_loss(
            outputs['logits'],
            target,
            outputs['confidence'],
            label_smoothing=args.label_smoothing
        )
        
        # L2 regularization
        l2_reg = 0
        for param in model.parameters():
            l2_reg += torch.norm(param)
        l2_reg *= args.l2_reg
        
        # Total loss
        loss = reconstruction_loss + dirichlet_loss + \
               args.lambda_cls * ce_loss + \
               args.lambda_cons * consistency_loss + \
               l2_reg
        
        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        prec_list, _, _ = accuracy(outputs['logits'], target, topk=(1,))
        prec1 = prec_list[0]  # Get the top-1 accuracy
        n = data.size(0)
        losses.update(loss.item(), n)
        top1.update(prec1.item(), n)
    
    return top1.avg, losses.avg

def validate(model, val_loader, criterion):
    model.eval()
    targets = []
    predictions = []
    confidence_correct = []
    confidence_incorrect = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.cuda(), target.cuda()
            outputs = model(data, is_training=False)
            
            # Store predictions and confidence
            _, pred = outputs['logits'].max(1)
            pred = pred.cpu()
            target_cpu = target.cpu()
            
            # Average confidence across spatial dimensions if needed
            confidence = outputs['confidence']
            if confidence.dim() > 2:
                confidence = confidence.mean(dim=(2, 3))
            confidence = confidence.cpu()
            
            # Track confidence for correct/incorrect predictions
            correct_mask = pred == target_cpu
            batch_confidence_correct = confidence[correct_mask].mean().item() if correct_mask.any() else 0.0
            batch_confidence_incorrect = confidence[~correct_mask].mean().item() if (~correct_mask).any() else 0.0
            
            if correct_mask.any():
                confidence_correct.append(batch_confidence_correct)
            if (~correct_mask).any():
                confidence_incorrect.append(batch_confidence_incorrect)
            
            targets.extend(target.cpu().tolist())
            predictions.extend(pred.tolist())
    
    # Compute metrics
    targets = np.array(targets)
    predictions = np.array(predictions)
    oa, aa_mean, kappa, aa = output_metric(targets, predictions)
    
    # Compute mean confidence
    mean_conf_correct = np.mean(confidence_correct) if confidence_correct else 0
    mean_conf_incorrect = np.mean(confidence_incorrect) if confidence_incorrect else 0
    
    return {
        'OA': oa,
        'AA': aa_mean,
        'Kappa': kappa,
        'confidence_correct': mean_conf_correct,
        'confidence_incorrect': mean_conf_incorrect
    }

def main():
    args = parse_args()
    
    # Set environment
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    if args.fix_random:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True
    
    # Prepare data
    train_loader, val_loader, test_loader, band, height, width, num_classes, _, _ = \
        prepare_dataset(args)
    
    if args.fast_debug:
        args.epoches = 5
        train_loader.dataset.data = train_loader.dataset.data[:100]
    
    # Create model
    model = S2VNetU(
        band=band,
        num_classes=num_classes,
        patch_size=args.patches,
        dropout_p=args.dropout_p,
        mc_samples_train=args.mc_samples_train,
        mc_samples_eval=args.mc_samples_eval
    ).cuda()
    
    # Setup training
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.AdamW(  # Switch to AdamW for better regularization
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup and cosine decay
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.epoches - warmup_epochs)))
        
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Create output directories
    os.makedirs('results', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epoches):
        # Train
        train_acc, train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, args)
        
        # Step scheduler
        scheduler.step()
        
        # Validate
        if (epoch + 1) % args.test_freq == 0:
            metrics = validate(model, val_loader, criterion)
            curr_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch+1:03d} "
                  f"Train Loss: {train_loss:.4f} "
                  f"Train Acc: {train_acc:.4f} "
                  f"Val OA: {metrics['OA']:.4f} "
                  f"Val AA: {metrics['AA']:.4f} "
                  f"LR: {curr_lr:.6f}")
            
            # Save best model
            if metrics['OA'] > best_acc:
                best_acc = metrics['OA']
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_acc': best_acc,
                }, f'checkpoints/s2vnet_u_best.pth')
        
        # Save final model
        if epoch == args.epoches - 1:
            # Evaluate on test set
            test_metrics = validate(model, test_loader, criterion)
            print("\nFinal Test Results:")
            print(f"OA: {test_metrics['OA']:.4f}")
            print(f"AA: {test_metrics['AA']:.4f}")
            print(f"Kappa: {test_metrics['Kappa']:.4f}")
            print(f"Mean Confidence (Correct): {test_metrics['confidence_correct']:.4f}")
            print(f"Mean Confidence (Incorrect): {test_metrics['confidence_incorrect']:.4f}")
            
            # Save metrics
            with open('results/summary.md', 'w') as f:
                f.write("# S2VNet-U Results\n\n")
                f.write("## Test Set Metrics\n\n")
                f.write(f"- Overall Accuracy: {test_metrics['OA']:.4f}\n")
                f.write(f"- Average Accuracy: {test_metrics['AA']:.4f}\n")
                f.write(f"- Kappa: {test_metrics['Kappa']:.4f}\n")
                f.write(f"- Mean Confidence (Correct): {test_metrics['confidence_correct']:.4f}\n")
                f.write(f"- Mean Confidence (Incorrect): {test_metrics['confidence_incorrect']:.4f}\n")

if __name__ == '__main__':
    main()
