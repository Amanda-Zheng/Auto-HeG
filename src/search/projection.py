import os
import sys

sys.path.insert(0, '../../../')
import numpy as np
import torch
import logging
import torch.utils
import src.utils_lib


def get_block_matrix(adj, y, soft_y):

    H1 = torch.mm(soft_y.t(), adj)
    H1 = torch.mm(H1, soft_y) / torch.mm(H1, torch.ones_like(soft_y))

    H2 = torch.mm(y.t(), adj)
    H2 = torch.mm(H2, y) / torch.mm(H2, torch.ones_like(y))

    crit = torch.dist(H1, H2, p=2)

    return crit

def infer_eval(data, model, criterion, ensem_param, device, log=True, test=False, train=False, weights_dict=None):
    objs = src.utils_lib.utils.AvgrageMeter()
    top1 = src.utils_lib.utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        if weights_dict is None:
            logits = model(data, weights_dict=None)
        else:
            logits = model(data, weights_dict=weights_dict)

    if train:
        input = logits[data.train_mask].to(device)
        target = data.y[data.train_mask].to(device)
        loss = criterion(input, target)
        if log:
            logging.info('train_loss={:.04f}'.format(loss.item()))
    if test:
        input = logits[data.test_mask].to(device)
        target = data.y[data.test_mask].to(device)
        loss = criterion(input, target)
        if log:
            logging.info('test_loss={:.04f}'.format(loss.item()))
    else:
        input = logits[data.val_mask].to(device)
        target = data.y[data.val_mask].to(device)
        loss = criterion(input, target)
        if log:
            logging.info('valid_loss={:.04f}'.format(loss.item()))
    prec1, prec5 = src.utils_lib.utils.accuracy(input, target, topk=(1, 3))
    n = data.val_mask.sum().item()
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    hetro_crit = get_block_matrix(data.nsadj,data.label_hot,logits)
    # comit sum of ensem crit
    ensem_crit = (1-ensem_param)*objs.avg + ensem_param*hetro_crit

    return top1.avg, objs.avg, hetro_crit, ensem_crit


def project_op(model, criterion, device, data, args, cell_type, test=False, train=False, selected_eid=None):
    ''' operation '''

    if cell_type == 'NA_PRIM':
        num_edges = model.na_alphas.shape[0]
        num_ops = model.na_alphas.shape[1]
    elif cell_type == 'SC_PRIM':
        num_edges = model.sc_alphas.shape[0]
        num_ops = model.sc_alphas.shape[1]
    elif cell_type == 'LA_PRIM':
        num_edges = model.la_alphas.shape[0]
        num_ops = model.la_alphas.shape[1]

    candidate_flags = model.candidate_flags[cell_type]
    proj_crit = args.proj_crit[cell_type]

    ## select an edge
    if selected_eid is None:
        remain_eids = torch.nonzero(candidate_flags).cpu().numpy().T[0]
        if args.edge_decision == "random":
            if cell_type == 'LA_PRIM' and len(remain_eids.tolist()) == 0:
                selected_eid = 0
            else:
                selected_eid = np.random.choice(remain_eids, size=1)[0]
            logging.info('Selected edge is {} in the cell type {}'.format(selected_eid, cell_type))

    #### select the best operation
    if proj_crit == 'loss':
        crit_idx = 1
        compare = lambda x, y: x > y
    elif proj_crit == 'acc':
        crit_idx = 0
        compare = lambda x, y: x < y
    elif proj_crit == 'hetro':
        crit_idx = 2
        compare = lambda x, y: x > y
    elif proj_crit == 'ensem_crit':
        crit_idx = 3
        compare = lambda x, y: x > y


    best_opid = 0
    crit_extrema = None
    for opid in range(num_ops):
        ## projection
        weights = model.get_projected_weights(cell_type)
        proj_mask = torch.ones_like(weights[selected_eid])
        proj_mask[opid] = 0
        weights[selected_eid] = weights[selected_eid] * proj_mask

        ## proj evaluation
        weights_dict = {cell_type: weights}
        # evaluating all operations in one edge
        valid_stats = infer_eval(data, model, criterion, args.ensem_param, device, test=test, train=train, weights_dict=weights_dict)
        crit = valid_stats[crit_idx]

        if crit_extrema is None or compare(crit, crit_extrema):
            crit_extrema = crit
            best_opid = opid
        logging.info('Operation mask and selection: valid_acc = {:.04f}, valid_loss= {:.04f}, hetero_crit = {:.04f}, ensem_crit = {:.04f}'.format(valid_stats[0],valid_stats[1],valid_stats[2],valid_stats[3]))

    #### project
    logging.info('Best opid = {} in the selected eid {}'.format(best_opid, selected_eid))
    return selected_eid, best_opid


def pt_project(data, model, criterion, device, args):
    model.eval()
    num_projs = model.num_edges
    tune_epochs = args.proj_intv * num_projs + 1
    proj_intv = args.proj_intv
    args.proj_crit = {'NA_PRIM': args.proj_crit_NA, 'SC_PRIM': args.proj_crit_SC, 'LA_PRIM': args.proj_crit_LA}

    start_epoch = 0

    ## projecting and tuning
    logging.info('Start projecting and tuning...')
    for epoch in range(start_epoch, tune_epochs):
        if epoch % proj_intv == 0 or epoch == tune_epochs - 1:
            if epoch < proj_intv * model.num_edges:
                logging.info('Conducting project op of {}'.format(epoch))
                if epoch < proj_intv * model.NA_num_edges:
                    logging.info('Conducting project op of {} for NA_PRIM'.format(epoch))
                    selected_eid_NA, best_opid_NA = project_op(model, criterion, device, data, args, cell_type='NA_PRIM',
                                                           test=False, train=False, selected_eid=None)
                    model.project_op(selected_eid_NA, best_opid_NA, cell_type='NA_PRIM')
                if epoch >= proj_intv * model.NA_num_edges and epoch< proj_intv * (model.SC_num_edges+model.NA_num_edges):
                    logging.info('Conducting project op of {} for SC_PRIM'.format(epoch))
                    selected_eid_SC, best_opid_SC = project_op(model, criterion, device, data, args,cell_type='SC_PRIM', test=False, train=False,selected_eid=None)
                    model.project_op(selected_eid_SC, best_opid_SC, cell_type='SC_PRIM')
                if epoch >= proj_intv * (model.SC_num_edges+model.NA_num_edges) and epoch < proj_intv * model.num_edges:
                    logging.info('Conducting project op of {} for LA_PRIM'.format(epoch))
                    selected_eid_LA, best_opid_LA = project_op(model, criterion, device, data, args,
                                                               cell_type='LA_PRIM', test=False, train=False,
                                                               selected_eid=None)
                    model.project_op(selected_eid_LA, best_opid_LA, cell_type='LA_PRIM')


    logging.info('Projection finished')
    genotype_pt = model.genotype_pt()
    logging.info('genotype_pt = {}'.format(genotype_pt))

    return genotype_pt
