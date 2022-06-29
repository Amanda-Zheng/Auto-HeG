import os
import os.path as osp
import numpy as np
import torch
import shutil
from collections import Counter
import logging
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.model_selection import StratifiedKFold
from torch import cat

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

def accuracy(output, target, topk):
  maxk = max(topk)
  batch_size = target.size(0)
  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  #logging.info('FLAG--{}--predition counter:{}'.format(flag, Counter(np.array(pred[:1].reshape(-1)))))
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    #if k==1:
    #    logging.info('FLAG--{}--This is the number of predict correct in per class:{}'.format(flag,Counter(np.array(target[correct[:k].reshape(-1)]))))
    correct_k = correct[:k].reshape(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

def accuracy_f(output, labels):

    preds = output.max(1)[1].type_as(labels)
    #logging.info('FLAG--{}--predition counter:{}'.format(flag, Counter(np.array(preds.reshape(-1)))))
    correct = preds.eq(labels).double()
    #logging.info('FLAG--{}--This is the number of predict correct in per class:{}'.format(flag, Counter(
    #    np.array(labels[preds.eq(labels)]))))
    correct = correct.sum()
    return correct / len(labels), labels, preds

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def save_checkpoint(state, is_best, save, per_epoch=False, prefix=''):
    filename = prefix
    if per_epoch:
        epoch = state['epoch']
        filename += 'checkpoint_{}.pth.tar'.format(epoch)
    else:
        filename += 'checkpoint.pth.tar'
    filename = os.path.join(save, filename)
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
  torch.save(model.state_dict(), model_path)

def load(model, model_path):
  model.load_state_dict(torch.load(model_path))

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def gen_uniform_40_40_20_split(data):
    skf = StratifiedKFold(10, shuffle=True, random_state=12345)
    idx = [torch.from_numpy(i) for _, i in skf.split(data.y, data.y)]
    return cat(idx[:4], 0), cat(idx[4:8], 0), cat(idx[8:], 0)

def save_load_split(data, gen_splits):

    split = gen_splits(data)
    data.train_mask = index_to_mask(split[0], data.num_nodes)
    data.val_mask = index_to_mask(split[1], data.num_nodes)
    data.test_mask = index_to_mask(split[2], data.num_nodes)

    return np.array(data.train_mask), np.array(data.val_mask), np.array(data.test_mask)

def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def mixhop_edge_info(data, train_args):
    data_edge_index_0 = torch.sparse_coo_tensor(data.edge_index, np.ones(data.edge_index.shape[1]))
    data_edge_index_2 = torch.sparse.mm(data_edge_index_0, data_edge_index_0)
    data_edge_index_3 = torch.sparse.mm(data_edge_index_0, data_edge_index_2)

    idx_0 = []
    for i_0 in range(data.edge_index.shape[1]):
        if data.edge_index[0, i_0] != data.edge_index[1, i_0]:
            idx_0.append(i_0)
    data_edge_index_new_0 = torch.index_select(data.edge_index, dim=-1, index=torch.from_numpy(np.array(idx_0)))
    data.edge_index = data_edge_index_new_0
    mix_edge_index = []
    mix_edge_index.append(data_edge_index_new_0)

    tmp_new_2 = torch.ge(data_edge_index_2.coalesce().values(), 2.0)
    data_edge_index_new_2_tmp = data_edge_index_2.coalesce().indices()[:,
                                torch.nonzero(data_edge_index_2.coalesce().values() * tmp_new_2).squeeze()]
    idx_2 = []
    for i_2 in range(data_edge_index_new_2_tmp.shape[1]):
        if data_edge_index_new_2_tmp[0, i_2] != data_edge_index_new_2_tmp[1, i_2]:
            idx_2.append(i_2)
    data_edge_index_new_2 = torch.index_select(data_edge_index_new_2_tmp, dim=-1,
                                               index=torch.from_numpy(np.array(idx_2)))
    mix_edge_index.append(data_edge_index_new_2)

    tmp_new_3 = torch.ge(data_edge_index_3.coalesce().values(), 3.0)
    data_edge_index_new_3_tmp = data_edge_index_3.coalesce().indices()[:,
                                torch.nonzero(data_edge_index_3.coalesce().values() * tmp_new_3).squeeze()]
    idx_3 = []
    for i_3 in range(data_edge_index_new_3_tmp.shape[1]):
        if data_edge_index_new_3_tmp[0, i_3] != data_edge_index_new_3_tmp[1, i_3]:
            idx_3.append(i_3)
    data_edge_index_new_3 = torch.index_select(data_edge_index_new_3_tmp, dim=-1,
                                               index=torch.from_numpy(np.array(idx_3)))
    mix_edge_index.append(data_edge_index_new_3)

    torch.save(mix_edge_index, './mixhop_info/mixhop_edge_index_' + train_args.data + '_' + str(train_args.num_layers))


def mixhop_edge_info_squirrel(data, train_args):
    data_edge_index_0 = torch.sparse_coo_tensor(data.edge_index, np.ones(data.edge_index.shape[1]))
    data_edge_index_2 = torch.sparse.mm(data_edge_index_0, data_edge_index_0)
    data_edge_index_3 = torch.sparse.mm(data_edge_index_0, data_edge_index_2)

    idx_0 = []
    for i_0 in range(data.edge_index.shape[1]):
        if data.edge_index[0, i_0] != data.edge_index[1, i_0]:
            idx_0.append(i_0)
    data_edge_index_new_0 = torch.index_select(data.edge_index, dim=-1, index=torch.from_numpy(np.array(idx_0)))
    data.edge_index = data_edge_index_new_0
    mix_edge_index = []
    mix_edge_index.append(data_edge_index_new_0)

    tmp_new_2 = torch.ge(data_edge_index_2.coalesce().values(), 4.0)
    data_edge_index_new_2_tmp = data_edge_index_2.coalesce().indices()[:,
                                torch.nonzero(data_edge_index_2.coalesce().values() * tmp_new_2).squeeze()]
    idx_2 = []
    for i_2 in range(data_edge_index_new_2_tmp.shape[1]):
        if data_edge_index_new_2_tmp[0, i_2] != data_edge_index_new_2_tmp[1, i_2]:
            idx_2.append(i_2)
    data_edge_index_new_2 = torch.index_select(data_edge_index_new_2_tmp, dim=-1,
                                               index=torch.from_numpy(np.array(idx_2)))
    mix_edge_index.append(data_edge_index_new_2)

    tmp_new_3 = torch.ge(data_edge_index_3.coalesce().values(), 5.0)
    data_edge_index_new_3_tmp = data_edge_index_3.coalesce().indices()[:,
                                torch.nonzero(data_edge_index_3.coalesce().values() * tmp_new_3).squeeze()]
    idx_3 = []
    for i_3 in range(data_edge_index_new_3_tmp.shape[1]):
        if data_edge_index_new_3_tmp[0, i_3] != data_edge_index_new_3_tmp[1, i_3]:
            idx_3.append(i_3)
    data_edge_index_new_3 = torch.index_select(data_edge_index_new_3_tmp, dim=-1,
                                               index=torch.from_numpy(np.array(idx_3)))
    mix_edge_index.append(data_edge_index_new_3)

    torch.save(mix_edge_index, './mixhop_info/mixhop_edge_index_' + train_args.data + '_' + str(train_args.num_layers))