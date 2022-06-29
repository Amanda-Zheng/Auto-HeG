import os
import sys
import time
import pickle
import torch
import src.utils_lib
import argparse
import torch.nn as nn
import torch.utils
import torch.backends.cudnn as cudnn
from sklearn.metrics import f1_score
import datetime
from os import path
from torch.autograd import Variable
from model import NetworkGNN as Network
from collections import Counter
from src.utils_lib import mixhop_edge_info

# from torch_geometric.data import DataLoader
from src.utils_lib.dataset_utils import DataLoader
from tensorboardX import SummaryWriter
from src.utils_lib.TDGNN_utils import *
import logging

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

def main(exp_args,i):
    global train_args
    train_args = exp_args

    tune_str = time.strftime('%Y%m%d-%H%M%S')
    train_save = train_args.log_dir + '/tune-{}-{}'.format(train_args.data,tune_str)
    writer = SummaryWriter(train_save + '/tbx_log_train4tune')

    if not os.path.exists(train_save):
        os.mkdir(train_save)

    global device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device(train_args.device)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        # sys.exit(1)

    np.random.seed(train_args.seed)
    # torch.cuda.set_device(train_args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(train_args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(train_args.seed)

    genotype = train_args.arch
    hidden_size = train_args.hidden_size

    dataset, data = DataLoader(train_args.data)
    if train_args.edge_index=='treecomp':
        treecom_edge_file = './tree_info/hop_edge_index_' + train_args.data + '_' + str(train_args.tree_layer)
        if (path.exists(treecom_edge_file) == False):
            edge_info(dataset, train_args)
        tree_edge_index = torch.load('./tree_info/hop_edge_index_' + train_args.data + '_' + str(train_args.tree_layer))
        data.tree_edge_index = tree_edge_index
        #hop_edge_att = torch.load('./tree_info/hop_edge_att_' + args.dataset + '_' + str(args.tree_layer))
    elif train_args.edge_index=='mixhop':
        mixhop_edge_file = './mixhop_info/mixhop_edge_index_' + train_args.data + '_' + str(train_args.num_layers)
        if (path.exists(mixhop_edge_file) == False):
            mixhop_edge_info(dataset[0], train_args)
        mix_edge_index = torch.load('./mixhop_info/mixhop_edge_index_' + train_args.data + '_' + str(train_args.num_layers))
        data.mix_edge_index = mix_edge_index
    else:
        idx_0 = []
        for i_0 in range(data.edge_index.shape[1]):
            if data.edge_index[0, i_0] != data.edge_index[1, i_0]:
                idx_0.append(i_0)
                #print(i_0, data.edge_index[:, i_0])
        #print(len(idx_0), data.edge_index.shape[1])
        data_edge_index_new_0 = torch.index_select(data.edge_index, dim=-1, index=torch.from_numpy(np.array(idx_0)))
        data.edge_index = data_edge_index_new_0

    # for searching, using fixed data split 0; but for finetuning, using all data splits.
    
    if train_args.kflag == 0:
        data_splits = np.load('splits_geom/' + train_args.data+ '_split_0.6_0.2_' + str(1) + '.npz')
    elif train_args.kflag==1:
        data_splits = np.load('splits_geom/' + train_args.data + '_split_0.6_0.2_' + str(i) + '.npz')
    
    logging.info('Data split using: {}'.format('splits_geom/' + train_args.data + '_split_0.6_0.2_' + str(i) + '.npz'))
    #data_splits = np.load('split/' + train_args.data + '/' + train_args.data + '_split_0.6_0.2_' + str(train_args.kfolds_idx) + '.npz')
    data.train_mask = torch.from_numpy(data_splits['train_mask']).bool()
    data.val_mask = torch.from_numpy(data_splits['val_mask']).bool()
    data.test_mask = torch.from_numpy(data_splits['test_mask']).bool()




    from collections import Counter
    print(Counter(np.array(data.y)))
    print(Counter(np.array(data.y[data.train_mask])))
    print(Counter(np.array(data.y[data.val_mask])))
    print(Counter(np.array(data.y[data.test_mask])))
    #assert False

    num_features = dataset.num_features
    num_classes = dataset.num_classes

    if train_args.weight_loss:

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
    criterion = criterion.to(device)

    # print(train_args.__dict__)
    # print(train_args.learning_rate)

    model = Network(genotype, criterion, num_features, num_classes, hidden_size,
                    num_layers=train_args.num_layers, in_dropout=train_args.in_dropout,
                    out_dropout=train_args.out_dropout, act=train_args.activation,
                    is_mlp=False, args=train_args)
    model = model.to(device)
    model.apply(weights_init)

    #print(model)
    logging.info("genotype={}, param size = {}MB, args={}".format(genotype, src.utils.count_parameters_in_MB(model),
                                                                  train_args.__dict__))

    if train_args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            train_args.learning_rate,
            # momentum=args.momentum,
            weight_decay=train_args.weight_decay
        )
    elif train_args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            train_args.learning_rate,
            momentum=train_args.momentum,
            weight_decay=train_args.weight_decay
        )
    elif train_args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            train_args.learning_rate,
            weight_decay=train_args.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(train_args.epochs),eta_min=train_args.learning_rate/5.)
    #print(get_num_hops(model))
    val_res = 0
    best_val_acc = best_test_acc = 0
    for epoch in range(train_args.epochs):
        train_acc, train_obj = train(train_args.data, data, model, criterion, optimizer,epoch)
        if train_args.cos_lr:
            scheduler.step()

        valid_acc, valid_obj = infer(train_args.data, data, model, criterion,epoch)
        test_acc, test_obj = infer(train_args.data, data, model, criterion, epoch, test=True)

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_test_acc = test_acc
        #logging.info(
        #    'epoch={}, lr={},valid_acc={:.04f}, test_acc={:.04f}'.format(
        #        epoch, scheduler.get_last_lr()[0], best_val_acc, best_test_acc))

        if epoch % 10 == 0:
            logging.info(
                'epoch={}, lr={}, train_obj={:.04f}, train_acc={:.04f}, valid_acc={:.04f}, test_acc={:.04f}'.format(
                    epoch, scheduler.get_last_lr()[0], train_obj, train_acc, best_val_acc, best_test_acc))
            # logging.info('epoch={}, train_acc={:.04f}, valid_acc={:.04f}, test_acc={:.04f},explore_num={}'.format(epoch, train_acc, valid_acc, test_acc, model.explore_num))
        # utils_lib.save(model, os.path.join(log_dir, 'weights.pt'))

        writer.add_scalar('Train/loss', train_obj, epoch)
        writer.add_scalar('Valid/loss', valid_obj, epoch)
        writer.add_scalar('Test/loss', test_obj, epoch)
        writer.add_scalar('Train/acc', train_acc, epoch)
        writer.add_scalar('Valid/acc', valid_acc, epoch)
        writer.add_scalar('Test/acc', valid_acc, epoch)


        src.utils.save(model, os.path.join(train_save, 'weights.pt'))

    return best_val_acc, best_test_acc, train_args


def train(dataset_name, data, model, criterion, optimizer,epoch):
    if dataset_name == 'PPI':
        return train_ppi(data, model, criterion, optimizer)
    else:
        return train_trans(data, model, criterion, optimizer,epoch)


def infer(dataset_name, data, model, criterion, epoch, test=False):
    if dataset_name == 'PPI':
        return infer_ppi(data, model, criterion, test=test)
    else:
        return infer_trans(data, model, criterion, epoch, test=test)


def train_trans(data, model, criterion, optimizer,epoch):
    objs = src.utils.AvgrageMeter()
    top1 = src.utils.AvgrageMeter()
    top5 = src.utils.AvgrageMeter()

    model.train()
    target = data.y[data.train_mask].to(device)

    optimizer.zero_grad()
    logits = model(data.to(device))

    input = logits[data.train_mask].to(device)

    loss = criterion(input, target)
    loss.backward()
    #nn.utils_lib.clip_grad_norm_(model.parameters(), train_args.grad_clip)
    optimizer.step()

    #prec1, prec5 = utils_lib.accuracy(input, target, topk=(1, 3))

    prec1, lbl_stat, pred_stat = src.utils.accuracy_f(input, target)

    if epoch % 10 == 0 and epoch >= 10:
        # print(epoch % 20, 21%20)
        #logging.info('FLAG--{}--predition counter:{}'.format('Train', Counter(np.array(pred_stat.reshape(-1).cpu()))))
        logging.info('FLAG--{}--This is the number of predict correct in per class:{}'.format('Train', Counter(
            np.array(lbl_stat[pred_stat.eq(lbl_stat)].cpu()))))

    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    #top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


def infer_trans(data, model, criterion, epoch, test=False):
    objs = src.utils.AvgrageMeter()
    top1 = src.utils.AvgrageMeter()
    top5 = src.utils.AvgrageMeter()
    model.eval()

    with torch.no_grad():
        logits = model(data.to(device))
    if test:
        mask = data.test_mask
        flag_in = 'Test'
    else:
        mask = data.val_mask
        flag_in = 'Val'
    input = logits[mask].to(device)
    target = data.y[mask].to(device)
    loss = criterion(input, target)

    #prec1, prec5 = utils_lib.accuracy(input, target, topk=(1, 3))
    prec1, lbl_stat, pred_stat = src.utils.accuracy_f(input, target)

    if epoch % 10 == 0 and epoch >= 10:
        #logging.info('FLAG--{}--predition counter:{}'.format(flag_in, Counter(np.array(pred_stat.reshape(-1).cpu()))))
        logging.info('FLAG--{}--This is the number of predict correct in per class:{}'.format(flag_in, Counter(
            np.array(lbl_stat[pred_stat.eq(lbl_stat)].cpu()))))

    n = data.val_mask.sum().item()
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    #top5.update(prec5.data.item(), n)

    return top1.avg, objs.avg


def train_ppi(data, model, criterion, optimizer):
    model.train()
    preds, ys = [], []
    total_loss = 0
    # input all data

    for train_data in data[0]:
        train_data = train_data.to(device)
        target = Variable(train_data.y).to(device)

        # train loss
        optimizer.zero_grad()
        input = model(train_data).to(device)
        loss = criterion(input, target)
        total_loss += loss.item()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), train_args.grad_clip)
        optimizer.step()

        preds.append((input > 0).float().cpu())
        ys.append(train_data.y.cpu())
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    prec1 = f1_score(y, pred, average='micro')
    # print('train_loss:', total_loss / len(data[0].dataset))
    return prec1, total_loss / len(data[0].dataset)


def infer_ppi(data, model, criterion, test=False):
    model.eval()
    total_loss = 0
    preds, ys = [], []
    if test:
        infer_data = data[2]
    else:
        infer_data = data[1]

    for val_data in infer_data:
        val_data = val_data.to(device)
        with torch.no_grad():
            logits = model(val_data).to(device)

        loss = criterion(logits, val_data.y.to(device))
        total_loss += loss.item()

        preds.append((logits > 0).float().cpu())
        ys.append(val_data.y.cpu())
    y, pred = torch.cat(ys, dim=0).numpy(), torch.cat(preds, dim=0).numpy()
    prec1 = f1_score(y, pred, average='micro')
    return prec1, total_loss / len(infer_data.dataset)

def run_fine_tune(args):
    tune_str = time.strftime('%Y%m%d-%H%M%S')
    lines = open(args.arch_filename, 'r').readlines()
    logging.info('Loading archs from {}'.format(args.arch_filename))
    test_res = []
    arch_set = set()
    argmax_arch_set = set()
    proj_loss_arch_set = set()
    proj_hetro_arch_set = set()
    proj_ensem_arch_set = set()
    suffix = args.arch_filename.split('_')[-1][:-4]

    for ind, l in enumerate(lines):
        try:
            logging.info('**********process {}-th/{}, logfilename={}**************'.format(ind+1, len(lines), os.path.join(log_dir, 'train4tune.log')))
            logging.info('**********process {}-th/{}**************'.format(ind+1, len(lines)))
            res = {}
            #iterate each searched architecture
            parts = l.strip().split(',')
            '''
            args.arch = parts[1].split('=')[1]
            res['searched_info'] = parts
            logging.info('here is the search info:{}'.format(parts))
            arch_set.add(args.arch)
            '''

            if args.arch_opt == 'argmax_arch':
                argmax_arch = parts[1].split('=')[1]
                args.arch = argmax_arch
                if argmax_arch in argmax_arch_set:
                    logging.info('ARGMAX_Selection: {}-th arch {} already searched...'.format(ind + 1, argmax_arch))
                    continue
                else:
                    argmax_arch_set.add(argmax_arch)
                res['searched_info'] = argmax_arch

            if args.arch_opt=='proj_loss_arch':
                proj_loss_arch = parts[2].split('=')[1]
                #proj_loss_arch = parts[0].split('=')[1]
                args.arch = proj_loss_arch
                if proj_loss_arch in proj_loss_arch_set:
                    logging.info('PROJ_LOSS_Selection: {}-th arch {} already searched...'.format(ind + 1, proj_loss_arch))
                    continue
                else:
                    proj_loss_arch_set.add(proj_loss_arch)
                res['searched_info'] = proj_loss_arch
            if args.arch_opt=='proj_hetro_arch':
                proj_hetro_arch = parts[3].split('=')[1]
                #proj_hetro_arch = parts[1].split('=')[1]
                args.arch = proj_hetro_arch
                if proj_hetro_arch in proj_hetro_arch_set:
                    logging.info('PROJ_HETRO_Selection: {}-th arch {} already searched...'.format(ind + 1, proj_hetro_arch))
                    continue
                else:
                    proj_hetro_arch_set.add(proj_hetro_arch)
                res['searched_info'] = proj_hetro_arch

            if args.arch_opt=='proj_ensem_arch':
                proj_ensem_arch = parts[4].split('=')[1]
                #proj_hetro_arch = parts[1].split('=')[1]
                args.arch = proj_ensem_arch
                if proj_ensem_arch in proj_ensem_arch_set:
                    logging.info('PROJ_HETRO_EMSEM_Selection: {}-th arch {} already searched...'.format(ind + 1, proj_ensem_arch))
                    continue
                else:
                    proj_ensem_arch_set.add(proj_ensem_arch)
                res['searched_info'] = proj_ensem_arch
            else:
                args.arch = args.arch
            test_accs=[]
            for i in range(args.kfolds):
                args.kflag = 1
                vali_acc, t_acc, test_args = main(args,i)
                logging.info('cal std: times:{}, valid_Acc:{:.04f}, test_acc:{:.04f}'.format(i,vali_acc,t_acc))
                test_accs.append(t_acc)

            test_accs = np.array(test_accs)
            logging.info('test_results_{}_times:{:.04f}+-{:.04f}'.format(args.kfolds, np.mean(test_accs), np.std(test_accs)))
            test_res.append(res)
            filepath = os.path.join(log_dir,'tuned_res')
            #filepath = os.path.join(log_dir, 'tuned_res/%s_res_%s_%s.pkl' % (args.data, tune_str, suffix))
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            fileout = filepath+'/%s_res_%s_%s.pkl' % (args.data, tune_str, suffix)
            with open(fileout, 'wb+') as fw:
                pickle.dump(test_res, fw)
            logging.info('**********finish {}-th/{}***************'.format(ind+1, len(lines)))
        except Exception as e:
            logging.info('error occured for {}-th, arch_info={}, error={}'.format(ind+1, l.strip(), e))
            import traceback
            traceback.print_exc()
    logging.info('finsh tunining {} argmax_archs, {} proj_loss_archs, {} proj_hetro_archs, {} proj_hetro_ensem_archs  for {} selection, saved in {}'.format(len(argmax_arch_set), len(proj_loss_arch_set), len(proj_hetro_arch_set), len(proj_ensem_arch_set), args.arch_opt, fileout))
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser("sane")
    parser.add_argument('--device', type=str, default='cpu', help='cuda:0 or cpu')
    parser.add_argument('--data', type=str, default='texas', help='location of the data corpus {cora,cornell,squirrel,wisconsin,texas,film}')
    parser.add_argument('--save_path', type=str, default='EXP_train', help='experiment name')
    parser.add_argument('--arch_filename', type=str,
                        default='',
                        help='given the location of searched res')
    parser.add_argument('--arch', type=str, default='gprgnn||gprgnn||cheb||skip||skip||l_max', help='given the specific of searched res')
    #parser.add_argument('--arch_opt', type=str, required=True, choices =['argmax_arch','proj_loss_arch','proj_hetro_arch','proj_ensem_arch','mix'], help='make option to evaluate which arch')
    parser.add_argument('--arch_opt', type=str, default='', choices =['argmax_arch','proj_loss_arch','proj_hetro_arch','proj_ensem_arch','mix'], help='make option to evaluate which arch')

    parser.add_argument('--optimizer', type=str, default='adam', help='optional optimizer')
    parser.add_argument('--activation', type=str, default='relu', help='optional activation')
    parser.add_argument('--learning_rate', type=float, default=0.007759968682108102)
    parser.add_argument('--weight_decay', type=float, default=0.00064170334262404)
    parser.add_argument('--in_dropout', type=float, default=0.6)
    parser.add_argument('--out_dropout', type=float, default=0.5)
    #parser.add_argument('--train_rate', type=float, default=0.025)
    #parser.add_argument('--val_rate', type=float, default=0.025)
    parser.add_argument('--tree_layer', type=int, default=10, help='tree layer for decomposition')
    parser.add_argument('--edge_index', type=str, default='mixhop', choices=['mixhop', 'treecomp', 'origin'],
                        help='edge_index')
    parser.add_argument('--num_layers', type=int, default=3, help='num of GNN layers in SANE')
    parser.add_argument('--hidden_size', type=int, default=128, help='num of hidden_size')
    parser.add_argument('--with_linear', action='store_true', default=False, help='whether to use linear in NaOp')
    parser.add_argument('--with_layernorm', action='store_true', default=False, help='whether to use layer norm')
    parser.add_argument('--epochs', type=int, default=1000, help='epoch in train GNNs.')
    parser.add_argument('--seed', type=int, default=2, help='random seed.')
    parser.add_argument('--grad_clip', type=int, default=5, help='grad_clip.')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum.')
    parser.add_argument('--kfolds', type=int, default=10, help='k-fold times of cross validation ')
    parser.add_argument('--cos_lr', action='store_true', default=False, help='using lr decay in training GNNs.')
    parser.add_argument('--self_loop', action='store_true', default=False, help='whether add self_loop.')
    parser.add_argument('--fix_last', action='store_true', default=True, help='fix last layer in design architectures.')
    parser.add_argument('--weight_loss', action='store_true', default=True, help='whether use weighted_cross_entropy loss')

    args = parser.parse_args()
    log_dir = './' + args.save_path + '/{}-{}'.format(args.data, datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    args.log_dir = log_dir
    log_format = '%(asctime)s %(message)s'
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'train4tune.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if args.arch_filename:
        run_fine_tune(args)

    elif args.arch:
        test_accs = []
        args.kflag =1
        for i in range(args.kfolds):
            vali_acc, t_acc, test_args = main(args,i)
            logging.info('cal std: times:{}, valid_Acc:{:.04f}, test_acc:{:.04f}'.format(i, vali_acc, t_acc))
            test_accs.append(t_acc)
        test_accs = np.array(test_accs)
        logging.info('test_results_5_times:{:.04f}+-{:.04f}'.format(np.mean(test_accs), np.std(test_accs)))

    #main()
