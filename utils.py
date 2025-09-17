import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F

class AvgrageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float32)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
           w = module.weight.data
           w.clamp_(1e-6, 1)


class FocalLossWithSmoothing(nn.Module):
  def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1, reduction='mean'):
    super(FocalLossWithSmoothing, self).__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.label_smoothing = label_smoothing
    self.reduction = reduction

  def forward(self, logits, targets):
    num_classes = logits.size(1)
    with torch.no_grad():
      smooth = self.label_smoothing
      true_dist = torch.zeros_like(logits)
      true_dist.fill_(smooth / (num_classes - 1))
      true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - smooth)

    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()

    pt = (true_dist * probs).sum(dim=1)
    focal_weight = (1 - pt).pow(self.gamma)
    alpha_weight = torch.ones_like(pt) * self.alpha

    loss = -(focal_weight * alpha_weight) * (true_dist * log_probs).sum(dim=1)

    if self.reduction == 'mean':
      return loss.mean()
    elif self.reduction == 'sum':
      return loss.sum()
    return loss
