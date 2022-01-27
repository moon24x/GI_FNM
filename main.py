import argparse
import os
import json
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from itertools import product
import copy
from sklearn.metrics import confusion_matrix
import datetime

from utils import *
from train_tools import *

from matching.pfnm import layer_wise_group_descent
from matching.pfnm import block_patching, patch_weights

from combine_nets import *
import pdb

def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def get_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--init_same', type=bool, required=False, help='All models init with same weight')

    # partitioning setting......
    parser.add_argument('--layers', type=int, required=True, help='do n_nets or n_layers')
    parser.add_argument('--n', nargs='+', type=int, required=True, help='the number of nets or layers')
    parser.add_argument('--n_layers', type=int, required=False, default=1, help="Number of hidden layers")
    parser.add_argument('--n_nets', type=int, required=False, default=10, help="Number of nets to initialize")
    parser.add_argument('--partition', type=str, required=False, help="Partition = homo/hetero/hetero-dir")
    parser.add_argument('--alpha', type=float, required=False, default=0.5,
                        help="Dirichlet distribution constant used for data partitioning")

    # save and load dir
    parser.add_argument('--loaddir', type=str, required=False, help='Load weights directory path')
    parser.add_argument('--logdir', type=str, required=False, help='Log directory path')

    # training setting......
    parser.add_argument('--dataset', type=str, required=False, default="mnist", help="Dataset [mnist/cifar10]")
    parser.add_argument('--datadir', type=str, required=False, default="./data/mnist", help="Data directory")
    parser.add_argument('--model', type=str, required=True, default="fcnet", help="which model to train")
    parser.add_argument('--net_config', type=lambda x: list(map(int, x.split(', '))), help="the layer config for Fully Connected neural network")
    parser.add_argument('--lr', type=float, required=False, default=0.01, help="Learning rate")
    parser.add_argument('--retrain_lr', type=float, default=0.1,
                        help='learning rate using in specific for local network retrain (default: 0.01)')
    parser.add_argument('--fine_tune_lr', type=float, default=0.1,
                        help='learning rate using in specific for fine tuning the softmax layer on the data center (default: 0.01)')
    parser.add_argument('--epochs', type=int, required=False, default=10, help="Epochs")
    parser.add_argument('--retrain_epochs', type=int, default=1,
                        help='how many epochs will be trained in during the locally retraining process')
    parser.add_argument('--dropout_p', type=float, required=False, default=0.0, help="Dropout probability. Default=0.0")
    parser.add_argument('--batch_size', type=int, required=False, default=64, help="the batch size to train")
    parser.add_argument('--reg', type=float, required=False, default=1e-6, help="L2 regularization strength")
    parser.add_argument('--retrain', type=bool, required=True, default=False,
                        help="Do we need retrain the init weights?")
    parser.add_argument('--train_prox', type=bool, required=False, default=False,
                        help="Do we train the network in prox way")

    # randomization setting
    parser.add_argument('--init_seed', type=int, required=False, default=0, help="Random seed")

    # matching experiment setting......
    parser.add_argument('--experiment', required=False, default="None", type=lambda s: s.split(','), help="Type of experiment to run. [none/w-ensemble/u-ensemble/pdm/all]")
    parser.add_argument('--trials', type=int, required=False, default=1, help="Number of trials for each run")
    parser.add_argument('--iter_epochs', type=int, required=False, default=5, help="Epochs for PDM-iterative method")
    parser.add_argument('--reg_fac', type=float, required=False, default=0.0, help="Regularization factor for PDM Iter")
    parser.add_argument('--pdm_sig', type=float, required=False, default=1.0, help="PDM sigma param")
    parser.add_argument('--pdm_sig0', type=float, required=False, default=1.0, help="PDM sigma0 param")
    parser.add_argument('--pdm_gamma', type=float, required=False, default=1.0, help="PDM gamma param")
    parser.add_argument('--lambdas', nargs='+', type=float, required=True, help='The list of KL coefficients')
    parser.add_argument('--communication_rounds', type=int, required=False, default=None, help="How many iterations of PDM matching should be done")
    parser.add_argument('--lr_decay', type=float, required=False, default=1.0, help="Decay LR after every PDM iterative communication")
    parser.add_argument('--fedavg_comm_round', type=int, default=19, 
                            help='how many round of communications we shoud use')
    
    # hardware setting (device and multiprocessing)
    parser.add_argument('--device', type=str, required=False, help="Device to run")
    parser.add_argument('--num_pool_workers', type=int, required=True, help='the num of workers')
    
    return parser

def trans_next_conv_layer_forward(layer_weight, next_layer_shape):
    reshaped = layer_weight.reshape(next_layer_shape).transpose((1, 0, 2, 3)).reshape((next_layer_shape[1], -1))
    return reshaped

def trans_next_conv_layer_backward(layer_weight, next_layer_shape):
    reconstructed_next_layer_shape = (next_layer_shape[1], next_layer_shape[0], next_layer_shape[2], next_layer_shape[3])
    reshaped = layer_weight.reshape(reconstructed_next_layer_shape).transpose(1, 0, 2, 3).reshape(next_layer_shape[0], -1)
    return reshaped

def train_net(net_id, net, train_dataloader, test_dataloader, args, reg_base_weights=None,
              save_path=None, device="cpu"):

    logger.info('Training network %s' % str(net_id))
    logger.info('n_training: %d' % len(train_dataloader))
    logger.info('n_test: %d' % len(test_dataloader))

    net.to(device)
    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: %f' % train_acc)
    logger.info('>> Pre-Training Test accuracy: %f' % test_acc)

    if args.model == "fcnet":
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                               lr=args.lr, weight_decay=0.0, amsgrad=True)      # L2_reg=0 because it's manually added later
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr)

    criterion = nn.CrossEntropyLoss().to(device)

    cnt = 0
    losses, running_losses = [], []

    global_weight_collector = []
    for param_idx, (key_name, param) in enumerate(net.state_dict().items()):
        global_weight_collector.append(param)

    for epoch in range(args.epochs):
        for batch_idx, (x, target) in enumerate(train_dataloader):
            x, target = x.to(device), target.to(device)
            #pdb.set_trace()
            l2_reg = torch.zeros(1)
            l2_reg.requires_grad = True
            l2_reg = l2_reg.to(device)

            optimizer.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = net(x)
            loss = criterion(out, target)

            if reg_base_weights is None:
                # Apply standard L2-regularization
                for param in net.parameters():
                    l2_reg = l2_reg + 0.5 * torch.pow(param, 2).sum()
            else:
                # Apply Iterative PDM regularization
                for pname, param in net.named_parameters():
                    if "bias" in pname:
                        continue

                    layer_i = int(pname.split('.')[1])

                    if pname.split('.')[2] == "weight":
                        weight_i = layer_i * 2
                        transpose = True

                    ref_param = reg_base_weights[weight_i]
                    ref_param = ref_param.T if transpose else ref_param

                    l2_reg = l2_reg + 0.5 * torch.pow((param - torch.from_numpy(ref_param).float()), 2).sum()
            #pdb.set_trace()
            #l2_reg = (reg * l2_reg).to(device)

            #########################we implement FedProx Here###########################
            if args.train_prox:
                fed_prox_reg = 0.0
                for param_index, param in enumerate(net.parameters()):
                    fed_prox_reg += ((0.001 / 2) * torch.norm((param - global_weight_collector[param_index])) ** 2)
                loss += fed_prox_reg
            ##############################################################################


            loss = loss + args.reg * l2_reg

            loss.backward()
            optimizer.step()

            cnt += 1
            losses.append(loss.item())

        logger.info('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), args.reg*l2_reg))

    train_acc = compute_accuracy(net, train_dataloader, device=device)
    test_acc, conf_matrix = compute_accuracy(net, test_dataloader, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy: %f' % train_acc)
    logger.info('>> Test accuracy: %f' % test_acc)
    logger.info(' ** Training complete **')
    return train_acc, test_acc

def local_train(nets, args, net_dataidx_map, device="cpu"):
    # save local dataset
    local_datasets = []
    local_train_accs = []
    local_test_accs = []

    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        logger.info("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        # move the model to cuda device:
        net.to(device)

        train_dl_local, test_dl_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
        weight_path = os.path.join(args.weight_dir, "model " + str(net_id) + ".pkl")
        local_datasets.append((train_dl_local, test_dl_local))
        if args.retrain:
            train_acc, test_acc = train_net(net_id, net, train_dl_local, test_dl_local, args, save_path=None, device=device)
            # saving the trained models here
            with open(weight_path, "wb") as f_:
                torch.save(net.state_dict(), f_)
        else:
            net.load_state_dict(torch.load(weight_path))
            train_acc = compute_accuracy(net, train_dl_local, device=device)
            test_acc, conf_matrix = compute_accuracy(net, test_dl_local, get_confusion_matrix=True, device=device)

        local_train_accs.append(train_acc)
        local_test_accs.append(test_acc)

    nets_list = list(nets.values())
    return nets_list, local_train_accs, local_test_accs


def local_retrain(local_datasets, weights, args, mode="bottom-up", freezing_index=0, ori_assignments=None,
                  device="cpu"):
    """
    freezing_index :: starting from which layer we update the model weights,
                      i.e. freezing_index = 0 means we train the whole network normally
                           freezing_index = len(model) means we freez the entire network
    """
    if args.dataset in ("cifar10", "cinic10"):
        input_channel = 3
    elif args.dataset == "mnist":
        input_channel = 1

    num_filters = [weights[0].shape[0], weights[2].shape[0]]
    input_dim = weights[4].shape[0]
    hidden_dims = [weights[4].shape[1], weights[6].shape[1]]
    matched_cnn = SimpleCNNContainer(input_channel=input_channel,
                                         num_filters=num_filters,
                                         kernel_size=5,
                                         input_dim=input_dim,
                                         hidden_dims=hidden_dims,
                                         output_dim=10)

    new_state_dict = {}
    model_counter = 0
    n_layers = int(len(weights) / 2)

    # we hardcoded this for now: will probably make changes later
    # if mode != "block-wise":
    if mode not in ("block-wise", "squeezing"):
        __non_loading_indices = []

    # handle the conv layers part which is not changing
    for param_idx, (key_name, param) in enumerate(matched_cnn.state_dict().items()):
        if "conv" in key_name or "features" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].reshape(param.size()))}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}
        elif "fc" in key_name or "classifier" in key_name:
            if "weight" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx].T)}
            elif "bias" in key_name:
                temp_dict = {key_name: torch.from_numpy(weights[param_idx])}

        new_state_dict.update(temp_dict)
    matched_cnn.load_state_dict(new_state_dict)

    for param_idx, param in enumerate(matched_cnn.parameters()):
        if mode == "bottom-up":
            # for this freezing mode, we freeze the layer before freezing index
            if param_idx < freezing_index:
                param.requires_grad = False
        elif mode == "per-layer":
            # for this freezing mode, we only unfreeze the freezing index
            if param_idx not in (2 * freezing_index - 2, 2 * freezing_index - 1):
                param.requires_grad = False

    matched_cnn.to(device).train()
    # start training last fc layers:
    train_dl_local = local_datasets[0]
    test_dl_local = local_datasets[1]

    if mode != "block-wise":
        if freezing_index < (len(weights) - 2):
            optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()),
                                            lr=args.retrain_lr, momentum=0.9)
        else:
            optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()),
                                            lr=(args.retrain_lr / 10), momentum=0.9, weight_decay=0.0001)
    else:
        # optimizer_fine_tune = optim.SGD(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=args.retrain_lr, momentum=0.9)
        optimizer_fine_tune = optim.Adam(filter(lambda p: p.requires_grad, matched_cnn.parameters()), lr=0.001,
                                         weight_decay=0.0001, amsgrad=True)

    criterion_fine_tune = nn.CrossEntropyLoss().to(device)

    logger.info('n_training: %d' % len(train_dl_local))
    logger.info('n_test: %d' % len(test_dl_local))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Pre-Training Training accuracy: %f' % train_acc)
    logger.info('>> Pre-Training Test accuracy: %f' % test_acc)

    if mode != "block-wise":
        if freezing_index < (len(weights) - 2):
            retrain_epochs = args.retrain_epochs
        else:
            retrain_epochs = int(args.retrain_epochs * 3)
    else:
        retrain_epochs = args.retrain_epochs

    for epoch in range(retrain_epochs):
        epoch_loss_collector = []
        for batch_idx, (x, target) in enumerate(train_dl_local):
            x, target = x.to(device), target.to(device)

            optimizer_fine_tune.zero_grad()
            x.requires_grad = True
            target.requires_grad = False
            target = target.long()

            out = matched_cnn(x)
            loss = criterion_fine_tune(out, target)
            epoch_loss_collector.append(loss.item())

            loss.backward()
            optimizer_fine_tune.step()

        # logger.info('Epoch: %d Loss: %f L2 loss: %f' % (epoch, loss.item(), reg*l2_reg))
        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        logger.info('Epoch: %d Epoch Avg Loss: %f' % (epoch, epoch_loss))

    train_acc = compute_accuracy(matched_cnn, train_dl_local, device=device)
    test_acc, conf_matrix = compute_accuracy(matched_cnn, test_dl_local, get_confusion_matrix=True, device=device)

    logger.info('>> Training accuracy after local retrain: %f' % train_acc)
    logger.info('>> Test accuracy after local retrain: %f' % test_acc)
    return matched_cnn

def BBP_MAP(nets_list, model_meta_data, layer_type, net_dataidx_map, traindata_cls_counts,
            averaging_weights, args, n_classes, sigma0=None, it=0, sigma=None, gamma=None,
            device="cpu", KL_reg=0, unlimi=False):
    # starting the neural matching
    models = nets_list
    cls_freqs = traindata_cls_counts
    assignments_list = []

    batch_weights = pdm_prepare_weights(models, device=device)
    raw_batch_weights = copy.deepcopy(batch_weights)

    logger.info("==" * 15)
    logger.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

    batch_freqs = pdm_prepare_freq(cls_freqs, n_classes)

    n_layers = int(len(batch_weights[0]) / 2)
    num_workers = len(nets_list)
    matching_shapes = []

    first_fc_index = None

    for layer_index in range(1, n_layers):
        layer_hungarian_weights, assignment, L_next = layer_wise_group_descent(
            batch_weights=batch_weights,
            layer_index=layer_index,
            sigma0_layers=sigma0,
            sigma_layers=sigma,
            batch_frequencies=batch_freqs,
            it=it,
            gamma_layers=gamma,
            model_meta_data=model_meta_data,
            model_layer_type=layer_type,
            n_layers=n_layers,
            matching_shapes=matching_shapes,
            args=args, KL_reg=KL_reg, unlimi=unlimi
        )
        assignments_list.append(assignment)

        # iii) load weights to the model and train the whole thing
        type_of_patched_layer = layer_type[2 * (layer_index + 1) - 2]
        if 'conv' in type_of_patched_layer or 'features' in type_of_patched_layer:
            l_type = "conv"
        elif 'fc' in type_of_patched_layer or 'classifier' in type_of_patched_layer:
            l_type = "fc"

        type_of_this_layer = layer_type[2 * layer_index - 2]
        type_of_prev_layer = layer_type[2 * layer_index - 2 - 2]
        first_fc_identifier = (('fc' in type_of_this_layer or 'classifier' in type_of_this_layer) and (
                    'conv' in type_of_prev_layer or 'features' in type_of_this_layer))

        if first_fc_identifier:
            first_fc_index = layer_index

        matching_shapes.append(L_next)
        tempt_weights = [
            ([batch_weights[w][i] for i in range(2 * layer_index - 2)] + copy.deepcopy(layer_hungarian_weights)) for w
            in range(num_workers)]

        # i) permutate the next layer wrt matching result
        for worker_index in range(num_workers):
            if first_fc_index is None:
                if l_type == "conv":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2],
                                                    L_next, assignment[worker_index],
                                                    layer_index + 1, model_meta_data,
                                                    matching_shapes=matching_shapes, layer_type=l_type,
                                                    dataset=args.dataset, network_name=args.model)
                elif l_type == "fc":
                    patched_weight = block_patching(batch_weights[worker_index][2 * (layer_index + 1) - 2].T,
                                                    L_next, assignment[worker_index],
                                                    layer_index + 1, model_meta_data,
                                                    matching_shapes=matching_shapes, layer_type=l_type,
                                                    dataset=args.dataset, network_name=args.model).T

            elif layer_index >= first_fc_index:
                patched_weight = patch_weights(batch_weights[worker_index][2 * (layer_index + 1) - 2].T, L_next,
                                               assignment[worker_index]).T

            tempt_weights[worker_index].append(patched_weight)

        # ii) prepare the whole network weights
        for worker_index in range(num_workers):
            for lid in range(2 * (layer_index + 1) - 1, len(batch_weights[0])):
                tempt_weights[worker_index].append(batch_weights[worker_index][lid])

        retrained_nets = []
        for worker_index in range(num_workers):
            dataidxs = net_dataidx_map[worker_index]
            train_dl_local, test_dl_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)

            logger.info("Re-training on local worker: {}, starting from layer: {}".format(worker_index,
                                                                                          2 * (layer_index + 1) - 2))
            retrained_cnn = local_retrain((train_dl_local, test_dl_local), tempt_weights[worker_index], args,
                                          freezing_index=(2 * (layer_index + 1) - 2), device=device)
            retrained_nets.append(retrained_cnn)
        batch_weights = pdm_prepare_weights(retrained_nets, device=device)

    ## we handle the last layer carefully here ...
    ## averaging the last layer
    matched_weights = []
    num_layers = len(batch_weights[0])

    with open('./matching_weights_cache/matched_layerwise_weights', 'wb') as weights_file:
        pickle.dump(batch_weights, weights_file)

    last_layer_weights_collector = []

    for i in range(num_workers):
        # firstly we combine last layer's weight and bias
        bias_shape = batch_weights[i][-1].shape
        last_layer_bias = batch_weights[i][-1].reshape((1, bias_shape[0]))
        last_layer_weights = np.concatenate((batch_weights[i][-2], last_layer_bias), axis=0)

        # the directed normalization doesn't work well, let's try weighted averaging
        last_layer_weights_collector.append(last_layer_weights)

    last_layer_weights_collector = np.array(last_layer_weights_collector)

    avg_last_layer_weight = np.zeros(last_layer_weights_collector[0].shape, dtype=np.float32)

    for i in range(n_classes):
        avg_weight_collector = np.zeros(last_layer_weights_collector[0][:, 0].shape, dtype=np.float32)
        for j in range(num_workers):
            avg_weight_collector += averaging_weights[j][i] * last_layer_weights_collector[j][:, i]
        avg_last_layer_weight[:, i] = avg_weight_collector

    # avg_last_layer_weight = np.mean(last_layer_weights_collector, axis=0)
    for i in range(num_layers):
        if i < (num_layers - 2):
            matched_weights.append(batch_weights[0][i])

    matched_weights.append(avg_last_layer_weight[0:-1, :])
    matched_weights.append(avg_last_layer_weight[-1, :])

    train_dl, test_dl = get_dataloader(args.dataset, args.datadir, args.batch_size, 32)
    train_acc, test_acc, _, _ = compute_full_cnn_accuracy(nets_list, matched_weights,
                                                          train_dl, test_dl, n_classes, device=device,
                                                          args=args)

    res = {}
    res['shapes'] = list(map(lambda x: x.shape, matched_weights))
    res['train_accuracy'] = train_acc
    res['test_accuracy'] = test_acc
    res['sigma0'] = sigma0
    res['sigma'] = sigma
    res['gamma'] = gamma
    res['weights'] = matched_weights

    return res

def fedavg_comm(batch_weights, model_meta_data, layer_type, net_dataidx_map,
                            averaging_weights, args,
                            train_dl_global,
                            test_dl_global,
                            comm_round=2,
                            device="cpu"):

    logger.info("=="*15)
    logger.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

    test_accuracy_comms = []
    for cr in range(comm_round):
        retrained_nets = []
        logger.info("Communication round : {}".format(cr))
        for worker_index in range(args.n_nets):
            dataidxs = net_dataidx_map[worker_index]
            train_dl_local, test_dl_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            
            # def local_retrain_fedavg(local_datasets, weights, args, device="cpu"):
            retrained_cnn = local_retrain_fedavg(len(dataidxs), (train_dl_local,test_dl_local), batch_weights[worker_index], args,
                                                 device=device)
            
            retrained_nets.append(retrained_cnn)
        batch_weights = pdm_prepare_weights(retrained_nets, device=device)

        total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
        averaged_weights = []
        num_layers = len(batch_weights[0])
        
        for i in range(num_layers):
            avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
            averaged_weights.append(avegerated_weight)

        train_acc, test_acc, _, _ = compute_full_cnn_accuracy(None, 
                                        averaged_weights,
                                        train_dl_global, 
                                        test_dl_global, 
                                        10, 
                                        device=device,
                                        args=args)

        batch_weights = [copy.deepcopy(averaged_weights) for _ in range(args.n_nets)]
        del averaged_weights
        test_accuracy_comms.append(test_acc)

    return test_accuracy_comms

def fedprox_comm(batch_weights, model_meta_data, layer_type, net_dataidx_map,
                            averaging_weights, args,
                            train_dl_global,
                            test_dl_global,
                            comm_round=2,
                            device="cpu"):

    logging.info("=="*15)
    logging.info("Weights shapes: {}".format([bw.shape for bw in batch_weights[0]]))

    test_accuracy_comms = []
    for cr in range(comm_round):
        retrained_nets = []
        logger.info("Communication round : {}".format(cr))
        for worker_index in range(args.n_nets):
            dataidxs = net_dataidx_map[worker_index]
            train_dl_local, test_dl_local = get_dataloader(args.dataset, args.datadir, args.batch_size, 32, dataidxs)
            
            # def local_retrain_fedavg(local_datasets, weights, args, device="cpu"):
            # local_retrain_fedprox(local_datasets, weights, mu, args, device="cpu")
            retrained_cnn = local_retrain_fedprox(len(dataidxs), (train_dl_local,test_dl_local), batch_weights[worker_index], mu=0.001,
                                                   args=args, device=device)
            
            retrained_nets.append(retrained_cnn)
        batch_weights = pdm_prepare_weights(retrained_nets, device=device)

        total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
        fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
        averaged_weights = []
        num_layers = len(batch_weights[0])
        
        for i in range(num_layers):
            avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
            averaged_weights.append(avegerated_weight)

        train_acc, test_acc, _, _ = compute_full_cnn_accuracy(None, 
                                        averaged_weights,
                                        train_dl_global, 
                                        test_dl_global, 
                                        10, 
                                        device=device,
                                        args=args)
        batch_weights = [copy.deepcopy(averaged_weights) for _ in range(args.n_nets)]
        del averaged_weights
        test_accuracy_comms.append(test_acc)

    return test_accuracy_comms

def load_new_state(nets, new_weights):

    for netid, net in nets.items():

        statedict = net.state_dict()
        weights = new_weights[netid]

        # Load weight into the network
        i = 0
        layer_i = 0

        while i < len(weights):
            weight = weights[i]
            i += 1
            bias = weights[i]
            i += 1

            statedict['layers.%d.weight' % layer_i] = torch.from_numpy(weight.T)
            statedict['layers.%d.bias' % layer_i] = torch.from_numpy(bias)
            layer_i += 1

        net.load_state_dict(statedict)

    return nets

def run_exp(n):
    print("Current n  is %d " % n)

    if args.device is not None:
        device_str = "cuda:" + str(args.device)
    else:
        device_str = "cuda:" + str(args.device)
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    print("Current device is :", device)

    if args.layers == 1:
        if args.model == "fcnet":
            if args.dataset == "mnist":
                args.net_config = list(map(int, ("784, " + "100, " * n + "10").split(', ')))
            elif args.dataset == "cifar10":
                args.net_config = list(map(int, ("3072, " + "100, " * n + "10").split(', ')))

        log_dir = os.path.join(args.logdir, "n_layers "+str(n))
        load_dir = os.path.join(args.loaddir, "n_layers "+str(n))
    else:
        args.n_nets = n
        log_dir = os.path.join(args.logdir, "n_nets "+str(n))
        load_dir = os.path.join(args.loaddir, "n_nets "+str(n))

    if not os.path.exists(log_dir):
        mkdirs(log_dir)

    logger.info("Experiment arguments: %s" % str(args))

    trials_res = {}
    trials_res["Experiment arguments"] = str(args)

    print("The total trials of n_nets %d is " % args.n_nets, args.trials)

    for trial in range(args.trials):
        #save_dir = os.path.join(log_dir, "trial "+str(trial))
        args.weight_dir = os.path.join(load_dir, "trial "+str(trial))
        if not os.path.exists(args.weight_dir):
            mkdirs(args.weight_dir)

        trials_res[trial] = {}
        seed = trial + args.init_seed
        trials_res[trial]['seed'] = seed
        print("Executing Trial %d " % trial)
        logger.info("#" * 100)
        logger.info("Executing Trial %d with seed %d" % (trial, seed))

        np.random.seed(seed)
        torch.manual_seed(seed)

        print("Partitioning data")
        X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
                        args.dataset, args.datadir, args.logdir, args.partition, args.n_nets, args.alpha)
        trials_res[trial]['Data statistics'] = str(traindata_cls_counts)
        n_classes = len(np.unique(y_train))

        averaging_weights = np.zeros((args.n_nets, n_classes), dtype=np.float32)

        for i in range(n_classes):
            total_num_counts = 0
            worker_class_counts = [0] * args.n_nets
            for j in range(args.n_nets):
                if i in traindata_cls_counts[j].keys():
                    total_num_counts += traindata_cls_counts[j][i]
                    worker_class_counts[j] = traindata_cls_counts[j][i]
                else:
                    total_num_counts += 0
                    worker_class_counts[j] = 0
            averaging_weights[:, i] = worker_class_counts / total_num_counts

        logger.info("averaging_weights: {}".format(averaging_weights))

        print("Initializing nets")
        nets, model_meta_data, layer_type = init_nets(args.net_config, args.dropout_p, args.n_nets, args)

        start = datetime.datetime.now()

        ### local training stage
        nets_list, local_train_accs, local_test_accs = local_train(nets, args, net_dataidx_map, device=device)

        end = datetime.datetime.now()
        timing = (end - start).seconds
        trials_res[trial]['loading time'] = timing

        trials_res[trial]['local_train_accs'] = local_train_accs
        trials_res[trial]['local_test_accs'] = local_test_accs
        train_dl, test_dl = get_dataloader(args.dataset, args.datadir, 32, 32)

        logger.info("*"*50)
        logger.info("Running experiments \n")

        nets_list = list(nets.values())

        trial_path = str(trial)

        if ("GI_FNM" in args.experiment) or ("all" in args.experiment):
            print("Computing hungarian matching")
            start = datetime.datetime.now()

            ## the KL penalty coefficients, 0 here stands for original PFNM
            ## for each trial, set some ranges of lambda to perform GI-FNM
            # lambdas = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
            lambdas = args.lambdas

            best_reg = 0
            comp = 0
          
            if not os.path.exists(log_dir):
                mkdirs(log_dir)

            for lam in lambdas:
            #for KL_reg in KL_regs:
                start = datetime.datetime.now()

                trials_res[trial]["lambda_ "+str(lam)] = {}

                if args.model == "fcnet":
                    res = compute_pdm_matching_multilayer(
                        nets_list, train_dl, test_dl, traindata_cls_counts, n_classes, it=args.iter_epochs,
                        sigma=args.pdm_sig,
                        sigma0=args.pdm_sig0, gamma=args.pdm_gamma, device=device, KL_reg=lam, unlimi=True
                    )
                else:
                    res = BBP_MAP(nets_list, model_meta_data, layer_type, net_dataidx_map,
                                  traindata_cls_counts, averaging_weights, args, n_classes, it=args.iter_epochs,
                                  sigma=args.pdm_sig, sigma0=args.pdm_sig0, gamma=args.pdm_gamma,
                                  device=device, KL_reg=lam, unlimi=True)

                end = datetime.datetime.now()

                logger.info("****** GI_FNM matching ******** ")
                logger.info("Sigma0: %s. sigma: %s gamma: %s. Test accuracy: %s. Train acc: %s. \n"
                          % (str(res['sigma0']), str(res['sigma']), str(res['gamma']), str(res['test_accuracy']), str(res['train_accuracy'])))
                # logger.info("PDM_KL log: %s " % str(res))

                trials_res[trial]["lambda_ "+str(lam)]["Result"] = {"Sigma0": res['sigma0'], "sigma": res['sigma'],
                                                         "gamma": res['gamma'], "Test accuracy": res['test_accuracy'],
                                                         "Train acc": res['train_accuracy']}
                trials_res[trial]["lambda_ "+str(lam)]["shape"] = res['shapes']

                logger.info("****** Save the matching weights ********")
                weights = res['weights']
                matching_weights_save_path = os.path.join(args.weight_dir, "lambda_ "+str(lam) + ".pkl")
                save_matching_weights(weights, args, matching_weights_save_path)
                # pdb.set_trace()
                trials_res[trial]["lambda_ "+str(lam)]["GI_FNM log"] = str(res)

                end = datetime.datetime.now()
                timing = (end - start).seconds
                trials_res[trial]["lambda_ "+str(lam)]["timing"] = timing

        if ("fedavg_comm" in args.experiment) or ("all" in args.experiment):
            print("Computing FedAvg communication accuracy")
            batch_weights = pdm_prepare_weights(nets_list, device=device)
            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
            logger.info("Total data points: {}".format(total_data_points))
            logger.info("Freq of FedAvg: {}".format(fed_avg_freqs))
            #pdb.set_trace()
            averaged_weights = []
            num_layers = len(batch_weights[0])
            for i in range(num_layers):
                avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
                averaged_weights.append(avegerated_weight)

            comm_init_batch_weights = [copy.deepcopy(averaged_weights) for _ in range(args.n_nets)]

            if args.model == "fcnet":
                train_acc, test_acc, _, _ = compute_pdm_net_accuracy(averaged_weights, 
                                        train_dl, 
                                        test_dl, 
                                        n_classes, 
                                        device=device)
                trials_res[trial]["fedavg (Test acc)"] = test_acc
            else:
                test_dice_coeff_comms = fedavg_comm(comm_init_batch_weights, model_meta_data, layer_type, 
                            net_dataidx_map, 
                            averaging_weights, args, 
                            train_dl,
                            test_dl,
                            comm_round=args.fedavg_comm_round,
                            device=device)
                trials_res[trial]["fedavg_comm (Test acc comms)"] = test_dice_coeff_comms
            
            trial_path += '_fedavg_comm'

        if ("fedprox_comm" in args.experiment) or ("all" in args.experiment):
            print("Computing FedProx communication accuracy")
            batch_weights = pdm_prepare_weights(nets_list, device=device)
            total_data_points = sum([len(net_dataidx_map[r]) for r in range(args.n_nets)])
            fed_avg_freqs = [len(net_dataidx_map[r]) / total_data_points for r in range(args.n_nets)]
            logger.info("Total data points: {}".format(total_data_points))
            logger.info("Freq of FedAvg: {}".format(fed_avg_freqs))

            averaged_weights = []
            num_layers = len(batch_weights[0])
            for i in range(num_layers):
                avegerated_weight = sum([b[i] * fed_avg_freqs[j] for j, b in enumerate(batch_weights)])
                averaged_weights.append(avegerated_weight)

            comm_init_batch_weights = [copy.deepcopy(averaged_weights) for _ in range(args.n_nets)]

            if args.model == "fcnet":
                train_acc, test_acc, _, _ = compute_pdm_net_accuracy(averaged_weights, 
                                        train_dl, 
                                        test_dl, 
                                        n_classes, 
                                        device=device)
                trials_res[trial]["fedprox (Test acc)"] = test_acc
            else:
                test_dice_coeff_comms = fedprox_comm(comm_init_batch_weights, model_meta_data, layer_type, 
                            net_dataidx_map, 
                            averaging_weights, args, 
                            train_dl,
                            test_dl,
                            comm_round=args.fedavg_comm_round,
                            device=device)
                trials_res[trial]["fedprox_comm (Test acc comms)"] = test_dice_coeff_comms

            trial_path += '_fedprox_comm'
                
        logger.info("Trial %d completed" % trial)
        logger.info("#"*100)

        trial_path += '.json'

        with open(os.path.join(log_dir, trial_path), 'w') as f:
            json.dump(trials_res[trial], f) 

    with open(os.path.join(log_dir, 'trials_res.json'), 'w') as f:
        json.dump(trials_res, f)

    return trials_res

abli_res = {}

from multiprocessing import Pool
import contextlib

z = lambda x: list(map(int, x.split(',')))
def abli_exp(n=[10], dataset="mnist"):

    partitions = ["hetero-dir"]
    #n_layer_list = [i+2 for i in range(n_layers-1)] # don't need layer_num 1
    if args.layers == 1:
        #n_list = [i+1 for i in range(n)]
        n_list = n
    else:
        #n_list = [i for i in range(0, n+1, 5)]
        #n_list[0] = 2
        n_list = n

    args.datadir = os.path.join("data", dataset)

    for partition in partitions:
        #logdir = os.path.join('log_abli', now.strftime("%Y-%m-%d %H"),
        #                      dataset, partition)
        
        logdir = os.path.join('GI_FNM_exp',
                              dataset, partition, args.model)
        args.loaddir = os.path.join('saved_weights', dataset, partition, args.model)
        args.logdir = logdir
        args.partition = partition
        print("Partition type is ", partition)
        abli_res = {}
        
        # set the sig, sig0, gamma,
        # this is referred by the sensitivity analysis in
        # http://proceedings.mlr.press/v97/yurochkin19a/yurochkin19a.pdf
        if dataset == "mnist" and partition == "homo":
            args.pdm_sig0 = 10
            args.pdm_sig = 0.5
            args.pdm_gamma = 10
        elif dataset == "mnist" and partition == "hetero-dir":
            args.pdm_sig0 = 3
            args.pdm_sig = 0.9
            args.pdm_gamma = 1
        elif dataset == "cifar10" and partition == "homo":
            args.pdm_sig0 = 10
            args.pdm_sig = 1
            args.pdm_gamma = 10
        elif dataset == "cifar10" and partition == "hetero-dir":
            args.pdm_sig0 = 10
            args.pdm_sig = 1
            args.pdm_gamma = 50

        with contextlib.closing(Pool(args.num_pool_workers)) as po:
            pool_results = po.map_async(run_exp, (n for n in n_list))
            results_list = pool_results.get()
        #run_exp(n_list[0])

if __name__ == "__main__":
    #run_exp()
    parser = get_parser()
    args = parser.parse_args()
    print("Abliation experiment running...")
    abli_exp(n=args.n, dataset=args.dataset)
