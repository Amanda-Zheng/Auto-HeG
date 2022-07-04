import os
import sys
import time
import torch
import src.utils_lib
import logging
import datetime
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from os import path
from src.utils_lib.TDGNN_utils import *
from collections import Counter

from torch.autograd import Variable
from src.search.model_search import Network, geno_map
from src.search.architect import Architect
from src.search.projection import pt_project
from src.utils_lib.utils import mixhop_edge_info

from src.utils_lib.dataset_utils import DataLoader, to_categorical
from tensorboardX import SummaryWriter
import scipy.sparse as sp
import copy

parser = argparse.ArgumentParser("AutoHeG-search wo shrink")
parser.add_argument('--data', type=str, default='texas', help='dataset')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.005, help='mininum learning rate')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--arch_learning_rate', type=float, default=0.0005, help='learning rate for arch learning')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch learning')
parser.add_argument('--tau_max', type=float, default=4, help='for gumbel softmax search gradient max value')
parser.add_argument('--tau_min', type=float, default=4, help='for gumbel softmax search gradient min value')

## epochs
parser.add_argument('--epochs_nspa', type=int, default=1000, help="shrinked new space re-search")
parser.add_argument('--epochs_tau_nspa', type=int, default=0, help="new space fix tau re-search")
parser.add_argument('--warm_up_epochs', type=int, default=0, help="per iteration drop num")

## network settings
parser.add_argument('--tree_layer', type=int, default=10, help='tree layer for decomposition')
parser.add_argument('--edge_index', type=str, default='mixhop', choices=['mixhop', 'treecomp', 'origin'],
                    help='edge_index')
parser.add_argument('--space_ver', type=str, default='v1', choices=['v0', 'v1', 'v2', 'v3'],
                    help='search space version, v0=all homo-SANE, v1= all homo & hetero, v2= subset of v1 with all homo and hetero, v3=all hetero')
parser.add_argument('--num_layers', type=int, default=3, help='num of GNN layers in AutoHeG')
parser.add_argument('--train_rate', type=float, default=0.4)
parser.add_argument('--val_rate', type=float, default=0.4)

parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--search_num', type=int, default=5, help='numbers of archi for seaching for onetime')
parser.add_argument('--device', type=str, default='cpu', help='cuda:0 or cpu')
parser.add_argument('--ensem_param', type=float, default=0.5,
                    help='ensembel crit for select arch params, range(0,1), close to 1, high weight for hetro and low for val_loss')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--transductive', action='store_true', help='use transductive settings in train_search.')
parser.add_argument('--with_conv_linear', action='store_true', default=False, help='in NAMixOp with linear op')
parser.add_argument('--fix_last', action='store_true', default=True, help='fix last layer in design architectures.')
parser.add_argument('--weight_loss', action='store_true', default=False, help='whether use weighted_cross_entropy loss')

## save
parser.add_argument('--save', type=str, default='EXP_search', help='experiment name')
parser.add_argument('--resume_cpk', type=str, default='', help="full expid to resume from, name == ckpt folder name")

## projection
parser.add_argument('--proj_crit_NA', type=str, default='hetro', choices=['loss', 'acc', 'hetro', 'ensem_crit'])
parser.add_argument('--proj_crit_SC', type=str, default='hetro', choices=['loss', 'acc', 'hetro', 'ensem_crit'])
parser.add_argument('--proj_crit_LA', type=str, default='hetro', choices=['loss', 'acc', 'hetro', 'ensem_crit'])
parser.add_argument('--edge_decision', type=str, default='random', choices=['random'], help='used for proj_op')
parser.add_argument('--proj_intv', type=int, default=1, help='interval between two projections')


args = parser.parse_args()

log_dir = './' + args.save + '/Arch-{}-{}-{}-{}'.format(args.edge_index, args.data, args.space_ver,
                                                        datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
writer = SummaryWriter(log_dir + '/tbx_log')
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(log_dir, 'search.log'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info('This is the log_dir: {}'.format(log_dir))


def main():
    global device
    device = torch.device(args.device)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        #sys.exit(1)

    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = {}".format(args.__dict__))

    # 1. loading dataset
    dataset, data = DataLoader(args.data)

    # 2. loading structure information with edge index, 'treecomp' from paper TDGNN, 'mixhop' is used in the proposed AutoHeG, else use original 'Adj'
    if args.edge_index == 'treecomp':
        treecom_edge_file = '../tree_info/hop_edge_index_' + args.data + '_' + str(args.tree_layer)
        if (path.exists(treecom_edge_file) == False):
            edge_info(dataset, args)
        tree_edge_index = torch.load('../tree_info/hop_edge_index_' + args.data + '_' + str(args.tree_layer))
        data.tree_edge_index = tree_edge_index
    elif args.edge_index == 'mixhop':
        mixhop_edge_file = '../mixhop_info/mixhop_edge_index_' + args.data + '_' + str(args.num_layers)
        if (path.exists(mixhop_edge_file) == False):
            mixhop_edge_info(dataset[0], args)
        mix_edge_index = torch.load('../mixhop_info/mixhop_edge_index_' + args.data + '_' + str(args.num_layers))
        data.mix_edge_index = mix_edge_index
        logging.info('This is the mix hop info: one-hop: {}, two-hop: {}, three-hop: {}' \
                     .format(mix_edge_index[0].shape, mix_edge_index[1].shape, mix_edge_index[2].shape))
    else:
        idx_0 = []
        for i_0 in range(data.edge_index.shape[1]):
            if data.edge_index[0, i_0] != data.edge_index[1, i_0]:
                idx_0.append(i_0)

        data_edge_index_new_0 = torch.index_select(data.edge_index, dim=-1, index=torch.from_numpy(np.array(idx_0)))
        data.edge_index = data_edge_index_new_0

    sadj = sp.coo_matrix(
        (np.ones(data.edge_index.t().shape[0]), (data.edge_index.t()[:, 0], data.edge_index.t()[:, 1])),
        shape=(data.num_nodes, data.num_nodes), dtype=np.float32)
    data.nsadj = torch.FloatTensor(sadj.todense())
    data.label_hot = torch.from_numpy(to_categorical(data.y))

    # 3. splitting dataset
    # NOTE: for searching, using fixed data split 4/4/2.

    data_splits = np.load('../splits_search/' + args.data + '/' + args.data + '_split_0.4_0.4_0.2.npz')
    data.train_mask = torch.from_numpy(data_splits['train_mask']).bool()
    data.val_mask = torch.from_numpy(data_splits['val_mask']).bool()
    data.test_mask = torch.from_numpy(data_splits['test_mask']).bool()

    # 4. whether using weighted loss
    if args.weight_loss:
        cnt_y = Counter(np.array(data.y))
        sum_cnt_y = sum(cnt_y.values())
        weight_loss = []
        for key in range(len(cnt_y.keys())):
            cnt_y[key] = 1 - (cnt_y[key] / sum_cnt_y)
            weight_loss.append(cnt_y[key])
        data.weight_loss = torch.from_numpy(np.array(weight_loss)).float()
        criterion = nn.CrossEntropyLoss(weight=data.weight_loss)
    else:
        criterion = nn.CrossEntropyLoss()

    # 5. initialization some settings and preparations for space shrinking of na_part
    criterion = criterion.to(device)
    PRIMITIVES = geno_map(args.space_ver)
    hidden_size = 32
    switches = []
    for i in range(3):  # 3 is the number of edges in na_part
        switches.append([True for j in range(len(PRIMITIVES))])
    switches_na = copy.deepcopy(switches)


    # 5. searching in the v1 whole search space without shrink
    model_new = Network(criterion, dataset.num_features, dataset.num_classes, hidden_size, device,
                        switches_na=switches_na,
                        args=args)
    model_new = model_new.to(device)
    logging.info("param size = {}MB".format(src.utils_lib.utils.count_parameters_in_MB(model_new)))
    network_params_new = []
    for k, v in model_new.named_parameters():
        if not (k.endswith('na_alphas') or k.endswith('sc_alphas') or k.endswith('la_alphas')):
            network_params_new.append(v)

    optimizer_new = torch.optim.Adam(
        network_params_new,
        args.learning_rate,
        weight_decay=args.weight_decay)
    scheduler_new = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_new, float(args.epochs_nspa), eta_min=args.learning_rate_min)

    arch_lr = args.arch_learning_rate
    search_cost = 0
    best_valid_acc = 0
    best_valid_obj = 0
    for epoch in range(args.epochs_nspa + args.epochs_tau_nspa):
        lr = scheduler_new.get_last_lr()[0]
        epoch_start = time.time()
        if epoch < args.epochs_nspa:
            model_new.set_tau(args.tau_max - (args.tau_max - args.tau_min) * epoch / (args.epochs_nspa - 1))
        else:
            model_new.set_tau(args.tau_min)
        if epoch < args.warm_up_epochs:
            train_acc, train_obj, arch_obj = train_shrink(epoch, data, model_new, network_params_new, criterion,
                                                          optimizer_new, lr, arch_lr, train_arch=False)
        else:

            train_acc, train_obj, arch_obj, architect_new = train_shrink(epoch, data, model_new, network_params_new,
                                                                         criterion,
                                                                         optimizer_new, lr, arch_lr, train_arch=True)
        logging.info('Epoch:{}, lr: {:.4f}, arch_lr:{:.4f}, geno:{}'.format(epoch, lr, arch_lr, model_new.genotype()))
        search_cost += (time.time() - epoch_start)
        valid_acc, valid_obj = infer_trans(epoch, data, model_new, criterion)
        test_acc, test_obj = infer_trans(epoch,data, model_new, criterion, test=True)

        if epoch == 0:
            best_valid_obj = valid_obj

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            save_state_dict = model_new.get_state_dict(epoch)
            src.utils_lib.utils.save_checkpoint(save_state_dict, False, log_dir, per_epoch=False,
                                                prefix=str(args.seed) + '_ACC_')

        if valid_obj < best_valid_obj:
            best_valid_obj = valid_obj
            save_state_dict = model_new.get_state_dict(epoch)
            src.utils_lib.utils.save_checkpoint(save_state_dict, False, log_dir, per_epoch=False,
                                                prefix=str(args.seed) + '_LOSS_')

        scheduler_new.step()
        logging.info('Training Epoch={}, Tau={:.04f}, '
                     'train_acc={:.04f}, valid_acc={:.04f}, test_acc={:.04f},'
                     'arch_loss = {:.04f}, train_loss={:.04f}, valid_loss={:.04f}, test_loss={:.04f}'
                     .format(epoch, model_new.get_tau(),
                             train_acc, valid_acc, test_acc,
                             arch_obj, train_obj, valid_obj, test_obj))

    logging.info('arch_param_magnitude:{}'.format(F.softmax(model_new.arch_parameters()[0], dim=-1)))

    argmax_geno, geno_out_loss, geno_out_hetro, geno_out_ensem = [], [], [], []

    if args.resume_cpk == '':
        filename = os.path.join(log_dir, str(args.seed) + '_LOSS_' + 'checkpoint.pth.tar')
        if os.path.isfile(filename):
            logging.info("=> loading projection checkpoint by best valid_loss cpk:'{}'".format(filename))
            checkpoint = torch.load(filename, map_location='cpu')
            model_new.set_state_dict(checkpoint)
            model_new.set_arch_parameters(checkpoint['alpha'])
            print(checkpoint['epoch'], checkpoint['alpha'])
        else:
            logging.info("=> no checkpoint found at '{}'".format(filename))
            exit(0)

        logging.info('Using heterophilic crit for arch selection...')
        args.proj_crit_NA, args.proj_crit_SC, args.proj_crit_SC = 'hetro', 'hetro', 'hetro'
        model_new._initialize_flags()
        geno_out_hetro_vloss = pt_project(data, model_new, criterion, device, args)
        geno_out_hetro.append(geno_out_hetro_vloss)

    return  geno_out_hetro


def logging_switches(switches, PRIMITIVES):
    for i in range(len(switches)):
        ops = []
        for j in range(len(switches[i])):
            if switches[i][j]:
                ops.append(PRIMITIVES[j])
        logging.info(ops)


def train_trans(data, model, architect, criterion, optimizer, lr, epoch):
    objs = src.utils_lib.utils.AvgrageMeter()
    top1 = src.utils_lib.utils.AvgrageMeter()

    model.train()
    target = Variable(data.y[data.train_mask], requires_grad=False).to(device)

    architect.step(data.to(device), lr, optimizer, unrolled=args.unrolled)

    logits = model(data.to(device))

    input = logits[data.train_mask].to(device)

    optimizer.zero_grad()
    loss = criterion(input, target)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, lbl_stat, pred_stat = src.utils_lib.utils.accuracy_f(input, target)

    if epoch % 20 == 0:
        logging.info('FLAG--{}--predition counter:{}'.format('Train', Counter(np.array(pred_stat.reshape(-1).cpu()))))
        logging.info('FLAG--{}--This is the number of predict correct in per class:{}'.format('Train', Counter(
            np.array(lbl_stat[pred_stat.eq(lbl_stat)].cpu()))))

    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)

    return top1.avg, objs.avg


def train_shrink(epoch, data, model, network_params, criterion, optimizer, lr, arch_lr, train_arch=False):
    objs = src.utils_lib.utils.AvgrageMeter()
    top1 = src.utils_lib.utils.AvgrageMeter()
    objs_arch = src.utils_lib.utils.AvgrageMeter()

    model.train()
    target = Variable(data.y[data.train_mask], requires_grad=False).to(device)

    if train_arch:
        architect = Architect(model, arch_lr, args)
        loss_arch = architect.step(data.to(device), lr, optimizer, unrolled=args.unrolled)
    else:
        loss_arch = torch.zeros(1)

    # finetune loss
    logits = model(data.to(device))
    input = logits[data.train_mask].to(device)

    optimizer.zero_grad()
    loss = criterion(input, target)
    loss.backward()
    optimizer.step()

    prec1, lbl_stat, pred_stat = src.utils_lib.utils.accuracy_f(input, target)

    if epoch % 20 == 0 and epoch >= 20:
        logging.info('FLAG--{}--predition counter:{}'.format('Train', Counter(np.array(pred_stat.reshape(-1).cpu()))))
        logging.info('FLAG--{}--This is the number of predict correct in per class:{}'.format('Train', Counter(
            np.array(lbl_stat[pred_stat.eq(lbl_stat)].cpu()))))

    n = input.size(0)
    objs.update(loss.data.item(), n)
    objs_arch.update(loss_arch.data.item(), n)
    top1.update(prec1.data.item(), n)
    if train_arch:
        return top1.avg, objs.avg, objs_arch.avg, architect
    else:
        return top1.avg, objs.avg, objs_arch.avg,


def infer_trans(epoch, data, model, criterion, test=False):
    objs = src.utils_lib.utils.AvgrageMeter()
    top1 = src.utils_lib.utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        logits = model(data.to(device))
    if test:
        input = logits[data.test_mask].to(device)
        target = data.y[data.test_mask].to(device)
        loss = criterion(input, target)
        flag_in = 'Test'
    else:
        input = logits[data.val_mask].to(device)
        target = data.y[data.val_mask].to(device)
        loss = criterion(input, target)
        flag_in = 'Val'

    prec1, lbl_stat, pred_stat = src.utils_lib.utils.accuracy_f(input, target)

    if epoch % 20 == 0 and epoch >= 20:
        logging.info('FLAG--{}--predition counter:{}'.format(flag_in, Counter(np.array(pred_stat.reshape(-1).cpu()))))
        logging.info('FLAG--{}--This is the number of predict correct in per class:{}'.format(flag_in, Counter(
            np.array(lbl_stat[pred_stat.eq(lbl_stat)].cpu()))))

    n = data.val_mask.sum().item()
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)

    return top1.avg, objs.avg


def get_min_k(input_in, k):
    input = copy.deepcopy(input_in)
    index = []
    for i in range(k):
        idx = np.argmin(input)
        index.append(idx)
        input[idx] = 1

    return index


def run_by_seed():
    res_vloss = []
    for i in range(args.search_num):
        logging.info('searched {}-th for {}...'.format(i + 1, args.data))
        seed = np.random.randint(0, 10000)
        args.seed = seed
        geno_out_hetro = main()

        res_vloss.append(
            'VLOSS_seed={}, proj_genotype_hetro={}, saved_dir={}'.format(
                args.seed, geno_out_hetro[0], log_dir))

    filename_vloss = log_dir + '/%s-searched_%s_res_best_valid_loss_arch.txt' % (args.data, time.strftime('%Y%m%d-%H%M%S'))
    fw_vloss = open(filename_vloss, 'w+')
    fw_vloss.write('\n'.join(res_vloss))
    fw_vloss.close()
    logging.info('searched res for {} saved in VLOSS:{}'.format(args.data, filename_vloss))

if __name__ == '__main__':
    run_by_seed()

