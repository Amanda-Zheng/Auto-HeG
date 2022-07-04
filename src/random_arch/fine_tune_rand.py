import os
import datetime
import time
import argparse
import pickle
import logging
import numpy as np
import sys

import hyperopt
from hyperopt import fmin, tpe, hp, Trials, partial, STATUS_OK
from src.finetune.train4tune import main

autoheg_space ={'model': 'AutoHeG',
         'hidden_size': hp.choice('hidden_size', [64]),
         'weight_decay': hp.choice("wr", [-5]),
         'optimizer': hp.choice('opt', ['adagrad']),
         'in_dropout': hp.choice('in_dropout', [3]),
         'out_dropout': hp.choice('out_dropout', [3]),
         'activation': hp.choice('act', ['relu'])
         }
def get_args():
    parser = argparse.ArgumentParser("AutoHeG-Finetune-random")
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument('--data', type=str, default='cornell', help='dataset')
    parser.add_argument('--save_path', type=str, default='EXP_train', help='experiment name')
    parser.add_argument('--arch_filename', type=str, default='./EXP_search/origin-cornell-20220315-124141-252046/Argmax-cornell-searched_res-20220315-124346-eps0.0-reg0.0005.txt', help='given the location of searched res')
    parser.add_argument('--arch', type=str, default='', help='given the specific of searched res')
    parser.add_argument('--arch_opt', type=str, required=True, choices =['random'], help='make option to evaluate which arch')
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--num_layers', type=int, default=3, help='num of GNN layers in AutoHeG')
    parser.add_argument('--fix_last', action='store_true', default=False, help='fix last layer in design architectures.')
    parser.add_argument('--tune_topK', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--record_time', action='store_true', default=False, help='whether to tune topK archs')
    parser.add_argument('--transductive', action='store_true', help='use transductive settings in train_search.')
    parser.add_argument('--with_linear', action='store_true', default=False, help='whether to use linear in NaOp')
    parser.add_argument('--with_layernorm', action='store_true', default=False, help='whether to use layer norm')
    parser.add_argument('--hyper_epoch', type=int, default=1, help='epoch in hyperopt.')
    parser.add_argument('--tree_layer', type=int, default=10, help='tree layer for decomposition')
    parser.add_argument('--edge_index', type=str, required=True, choices=['mixhop', 'treecomp','origin'],
                        help='edge_index of graph structures')
    parser.add_argument('--epochs', type=int, default=400, help='epoch in finetune GNNs.')
    parser.add_argument('--kfolds', type=int, default=10, help='k-folds cross validation')
    parser.add_argument('--cos_lr', action='store_true', default=False, help='using lr decay in training GNNs.')


    global args1
    args1 = parser.parse_args()

class ARGS(object):

    def __init__(self):
        super(ARGS, self).__init__()

def generate_args(arg_map):

    args = ARGS()
    for k, v in arg_map.items():
        setattr(args, k, v)
    setattr(args, 'rnd_num', 1)
    args.learning_rate = args1.learning_rate
    args.weight_decay = 10**args.weight_decay
    args.in_dropout = args.in_dropout / 10.0
    args.out_dropout = args.out_dropout / 10.0
    args.data = args1.data
    args.kfolds = args1.kfolds
    args.epochs = args1.epochs
    args.arch = args1.arch
    args.arch_opt = args1.arch_opt
    args.device = args1.device
    args.num_layers = args1.num_layers
    args.seed = 2
    args.grad_clip = 5
    args.momentum = 0.9
    args.with_linear = args1.with_linear
    args.with_layernorm = args1.with_layernorm
    args.transductive = args1.transductive
    args.cos_lr = args1.cos_lr
    args.fix_last = args1.fix_last
    args.log_dir = log_dir
    args.edge_index = args1.edge_index
    args.tree_layer = args1.tree_layer
    args.kflag = 0

    return args

def objective(args):
    args = generate_args(args)
    vali_acc, test_acc, args = main(args,0)
    return {
        'loss': -vali_acc,
        'test_acc': test_acc,
        'status': STATUS_OK,
        'eval_time': round(time.time(), 2),
        }

def run_fine_tune():
    tune_str = time.strftime('%Y%m%d-%H%M%S')
    lines = open(args1.arch_filename, 'r').readlines()
    logging.info('Loading archs from {}'.format(args1.arch_filename))

    suffix = args1.arch_filename.split('_')[-1][:-4]
    test_res = []
    arch_set = set()

    if args1.data == 'CiteSeer':
        autoheg_space['learning_rate'] = hp.uniform("lr", -2.5, -1.6)
        autoheg_space['weight_decay'] = hp.choice('wr', [-8])
        autoheg_space['in_dropout'] = hp.choice('in_dropout', [5])
        autoheg_space['out_dropout'] = hp.choice('out_dropout', [0])

    for ind, l in enumerate(lines):
        try:
            logging.info('**********process {}-th/{}, logfilename={}**************'.format(ind+1, len(lines), os.path.join(log_dir, 'finetune.log')))
            logging.info('**********process {}-th/{}**************'.format(ind+1, len(lines)))
            res = {}
            #iterate each searched architecture
            parts = l.strip().split(',')

            args1.arch = parts[1].split('=')[1]
            res['searched_info'] = parts
            logging.info('here is the search info:{}'.format(parts))
            arch_set.add(args1.arch)

            start = time.time()
            trials = Trials()
            #tune with validation acc, and report the test accuracy with the best validation acc
            best = fmin(objective, autoheg_space, algo=partial(tpe.suggest, n_startup_jobs=int(args1.hyper_epoch/5)),
                        max_evals=args1.hyper_epoch, trials=trials)

            space = hyperopt.space_eval(autoheg_space, best)
            logging.info('best space is {}'.format(space))
            res['best_space'] = space
            args = generate_args(space)
            logging.info('best args from space is {}'.format(args.__dict__))
            res['tuned_args'] = args.__dict__

            record_time_res = []
            c_vali_acc, c_test_acc = 0, 0
            #report the test acc with the best vali acc
            for d in trials.results:
                if -d['loss'] > c_vali_acc:
                    c_vali_acc = -d['loss']
                    c_test_acc = d['test_acc']
                    record_time_res.append('%s,%s,%s' % (d['eval_time'] - start, c_vali_acc, c_test_acc))
            res['test_acc'] = c_test_acc
            logging.info('test_acc={:.04f}'.format(c_test_acc))

            test_accs=[]

            for i in range(args1.kfolds):
                args.kflag = 1
                vali_acc, t_acc, test_args = main(args,i)
                logging.info('cal std: times:{}, valid_Acc:{:.04f}, test_acc:{:.04f}'.format(i,vali_acc,t_acc))
                test_accs.append(t_acc)

            test_accs = np.array(test_accs)
            logging.info('test_results_{}_times:{:.04f}+-{:.04f}'.format(args1.kfolds, np.mean(test_accs), np.std(test_accs)))
            test_res.append(res)
            filepath = os.path.join(log_dir,'tuned_res')
            if not os.path.exists(filepath):
                os.mkdir(filepath)
            fileout = filepath+'/%s_res_%s_%s.pkl' % (args1.data, tune_str, suffix)
            with open(fileout, 'wb+') as fw:
                pickle.dump(test_res, fw)
            logging.info('**********finish {}-th/{}***************'.format(ind+1, len(lines)))
        except Exception as e:
            logging.info('error occured for {}-th, arch_info={}, error={}'.format(ind+1, l.strip(), e))
            import traceback
            traceback.print_exc()
    logging.info('finsh tunining {} archs {} selection, saved in {}'.format(len(arch_set),args1.arch_opt, fileout))


if __name__ == '__main__':
    get_args()
    log_dir = './' + args1.save_path + '/{}-{}-{}'.format(args1.arch_opt,args1.data,datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    fh = logging.FileHandler(os.path.join(log_dir, 'finetune.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    if args1.arch_filename:
        run_fine_tune()

