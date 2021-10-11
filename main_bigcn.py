# -*- coding: utf-8 -*-
import sys,os
import argparse
from time import time
sys.path.append(os.getcwd())
from Process.process import *
import torch
import torch.nn.functional as F
import numpy as np
from tools.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from Process.rand5fold import *
from tools.evaluate import *
from model.BiGCN import BiGCN

"""
The Bi-GCN is referred from the original implementation from the paper authors 

Rumor Detection on Social Media with Bi-Directional Graph Convolutional Networks

Link: https://arxiv.org/pdf/2001.06362.pdf
Code: https://github.com/TianBian95/BiGCN 
"""

def train_model(treeDic, x_test, x_train, args, iter):
    model = BiGCN(args).to(args.device)

    TD_params = list(map(id,model.TDrumorGCN.conv1.parameters()))
    TD_params += list(map(id,model.TDrumorGCN.conv2.parameters()))
    BU_params = list(map(id,model.BUrumorGCN.conv1.parameters()))
    BU_params += list(map(id, model.BUrumorGCN.conv2.parameters()))
    base_params=filter(lambda p:id(p) not in BU_params+TD_params, model.parameters())
    optimizer = torch.optim.Adam([
        {'params':base_params},
        {'params':model.BUrumorGCN.conv1.parameters(),'lr':args.lr / args.lr_scale_bu},
        {'params': model.BUrumorGCN.conv2.parameters(), 'lr': args.lr / args.lr_scale_bu},
        {'params': model.TDrumorGCN.conv1.parameters(), 'lr': args.lr / args.lr_scale_td},
        {'params': model.TDrumorGCN.conv2.parameters(), 'lr': args.lr / args.lr_scale_td}
    ], lr=args.lr, weight_decay=args.l2)

    model.train()
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    for epoch in range(args.n_epochs):
        traindata_list, testdata_list = loadData(args.datasetname, treeDic, x_train, x_test, args.TDdroprate, args.BUdroprate)
        train_loader = DataLoader(traindata_list, batch_size=args.batchsize, shuffle=True, num_workers=10)
        test_loader = DataLoader(testdata_list, batch_size=args.batchsize, shuffle=False, num_workers=10)
        avg_loss = []
        avg_acc = []
        batch_idx = 0
        for Batch_data in train_loader:
            Batch_data.to(args.device)
            out_labels = model(Batch_data)
            finalloss = F.nll_loss(out_labels,Batch_data.y)
            loss=finalloss
            optimizer.zero_grad()
            loss.backward()
            avg_loss.append(loss.item())
            optimizer.step()
            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(Batch_data.y).sum().item()
            train_acc = correct / len(Batch_data.y)
            avg_acc.append(train_acc)
            print("Iter {:03d} | Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(iter,epoch, batch_idx,
                                                                                                 loss.item(),
                                                                                                 train_acc))
            batch_idx = batch_idx + 1

        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))

        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval()
        for Batch_data in test_loader:
            Batch_data.to(args.device)
            val_out = model(Batch_data)
            val_loss  = F.nll_loss(val_out, Batch_data.y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(Batch_data.y).sum().item()
            val_acc = correct / len(Batch_data.y)
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, Batch_data.y)
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f}| Val_Accuracy {:.4f}".format(epoch, np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))

        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
        early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                       np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'BiGCN', args.datasetname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return accs,F1,F2,F3,F4



def init_seeds(seed=2020):
    torch.manual_seed(seed)  # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed)  # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed)  # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print("Init_seeds....", seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--datasetname',type=str, default="Twitter16", metavar='dataname', help='dataset name')
    parser.add_argument('--input_features', type=int, default=5000, metavar='inputF', help='dimension of input features (TF-IDF)')
    parser.add_argument('--hidden_features', type=int, default=64, metavar='hidden_features', help='dimension of graph hidden state')
    parser.add_argument('--output_features', type=int, default=64, metavar='output_features', help='dimension of output features')
    parser.add_argument('--num_class', type=int, default=4, metavar='numclass', help='number of classes')
    parser.add_argument('--seed', type=int, default=2020, help='random state seed')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='does not use GPU')
    parser.add_argument('--num_cuda', type=int, default=1, help='index of GPU')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR', help='learning rate')
    parser.add_argument('--lr_scale_bu', type=int, default=5, metavar='LRSB', help='learning rate scale for bottom-up direction')
    parser.add_argument('--lr_scale_td', type=int, default=1, metavar='LRST', help='learning rate scale for top-down direction')
    parser.add_argument('--l2', type=float, default=1e-4, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')
    parser.add_argument('--patience', type=int, default=10, metavar='patience', help='patience for early stop')
    parser.add_argument('--batchsize', type=int, default=128, metavar='BS', help='batch size')
    parser.add_argument('--n_epochs', type=int, default=200, metavar='E', help='number of max epochs')
    parser.add_argument('--iterations', type=int, default=1, metavar='F', help='number of iterations for 5-fold cross-validation')
    parser.add_argument('--TDdroprate', type=float, default=0.2, metavar='TDdroprate', help='drop rate for edges in the top-down propagation graph')
    parser.add_argument('--BUdroprate', type=float, default=0.2, metavar='TDdroprate', help='drop rate for edges in the bottom-up dispersion graph')

    args = parser.parse_args()


    if not args.no_cuda:
        print('Running on GPU:{}'.format(args.num_cuda))
        args.device = torch.device('cuda:{}'.format(args.num_cuda) if torch.cuda.is_available() else 'cpu')
    else:
        print('Running on CPU')
        args.device = torch.device('cpu')
    print(args)

    init_seeds(seed=args.seed)
        
    total_accs, total_NR_F1, total_FR_F1, total_TR_F1, total_UR_F1 = [], [], [], [], []
    treeDic = loadTree(args.datasetname)

    for iter in range(args.iterations):
        iter_timestamp = time()
        fold_tests, fold_trains = load5foldData(args.datasetname, seed=args.seed)

        accs, NR_F1,  FR_F1, TR_F1, UR_F1 = [], [], [], [], []
        for fold_idx in range(5):
            fold_timestamp = time()
            acc, F1, F2, F3, F4 = train_model(treeDic, fold_tests[fold_idx], fold_trains[fold_idx], args, iter)
            accs.append(acc)
            NR_F1.append(F1)
            FR_F1.append(F2)
            TR_F1.append(F3)
            UR_F1.append(F4)

            print("Iter:{}/{}\tFold:{}/5 - Acc:{:.4f}\tNR_F1:{:.4f}\tFR_F1:{:.4f}\tTR_F1:{:.4f}\tUR_F1:{:.4f}\tTime:{:.4f}s".format(iter, args.iterations,
                                                                                                                                    fold_idx,
                                                                                                                                    acc,F1,F2,F3,F4,
                                                                                                                                    time()-fold_timestamp))


    

        iter_acc = np.mean(accs)
        iter_NR_F1 = np.mean(NR_F1)
        iter_FR_F1 = np.mean(FR_F1)
        iter_TR_F1 = np.mean(TR_F1)
        iter_UR_F1 = np.mean(UR_F1)

        print("****  Iteration Result {}/{} Time:{:.4f}s  ****".format(iter, args.iterations, time() - iter_timestamp))
        print("Acc:{:.4f}\tNR_F1:{:.4f}\tFR_F1:{:.4f}\tTR_F1:{:.4f}\tUR_F1:{:.4f}\t\tavg_F1:{:.4f}".format(np.mean(accs), np.mean(NR_F1), np.mean(FR_F1), np.mean(TR_F1), np.mean(UR_F1),
                                                                                                           (np.mean(NR_F1)+ np.mean(FR_F1)+np.mean(TR_F1)+np.mean(UR_F1))/4
                                                                                                           ))


    print("****  Total Result  ****")
    print("Acc:{:.4f}\tNR_F1:{:.4f}\tFR_F1:{:.4f}\tTR_F1:{:.4f}\tUR_F1:{:.4f}".format(np.mean(iter_acc), np.mean(iter_NR_F1), np.mean(iter_FR_F1), np.mean(iter_TR_F1), np.mean(iter_UR_F1)))


