import sys,os
sys.path.append(os.getcwd())
import torch as th
from torch_scatter import scatter_mean
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import copy
from torch.nn import BatchNorm1d
from collections import OrderedDict


class TDrumorGCN(th.nn.Module):
    def __init__(self,args):
        super(TDrumorGCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(args.input_features, args.hidden_features)
        self.conv2 = GCNConv(args.input_features + args.hidden_features, args.output_features)
        self.device = args.device

        self.num_features_list = [args.hidden_features * r for r in [1]]

        def creat_network(self, name):
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                layer_list[name + 'conv{}'.format(l)] = th.nn.Conv1d(
                    in_channels=args.hidden_features,
                    out_channels=args.hidden_features,
                    kernel_size=1,
                    bias=False)
                layer_list[name + 'norm{}'.format(l)] = th.nn.BatchNorm1d(num_features=args.hidden_features)
                layer_list[name + 'relu{}'.format(l)] = th.nn.LeakyReLU()
            layer_list[name + 'conv_out'] = th.nn.Conv1d(in_channels=args.hidden_features,
                                                         out_channels=1,
                                                         kernel_size=1)
            return layer_list

        self.sim_network = th.nn.Sequential(creat_network(self, 'sim_val'))
        mod_self = self
        mod_self.num_features_list = [args.hidden_features]
        self.W_mean = th.nn.Sequential(creat_network(mod_self, 'W_mean'))
        self.W_bias = th.nn.Sequential(creat_network(mod_self, 'W_bias'))
        self.B_mean = th.nn.Sequential(creat_network(mod_self, 'B_mean'))
        self.B_bias = th.nn.Sequential(creat_network(mod_self, 'B_bias'))
        self.fc1 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        self.fc2 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        self.dropout = th.nn.Dropout(args.dropout)
        self.eval_loss = th.nn.KLDivLoss(reduction='batchmean')
        self.bn1 = BatchNorm1d(args.hidden_features + args.input_features)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        if self.args.edge_infer_td:
            edge_loss, edge_pred = self.edge_infer(x, edge_index)
        else:
            edge_loss, edge_pred = None, None

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x, edge_index, edge_weight=edge_pred)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x, root_extend), 1)

        x = scatter_mean(x, data.batch, dim=0)
        return x, edge_loss

    def edge_infer(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        x_i = x[row - 1].unsqueeze(2)
        x_j = x[col - 1].unsqueeze(1)
        x_ij = th.abs(x_i - x_j)
        sim_val = self.sim_network(x_ij)
        edge_pred = self.fc1(sim_val)
        edge_pred = th.sigmoid(edge_pred)
        w_mean = self.W_mean(x_ij)
        w_bias = self.W_bias(x_ij)
        b_mean = self.B_mean(x_ij)
        b_bias = self.B_bias(x_ij)
        logit_mean = w_mean * sim_val + b_mean
        logit_var = th.log((sim_val ** 2) * th.exp(w_bias) + th.exp(b_bias))
        edge_y = th.normal(logit_mean, logit_var)
        edge_y = th.sigmoid(edge_y)
        edge_y = self.fc2(edge_y)
        logp_x = F.log_softmax(edge_pred, dim=-1)
        p_y = F.softmax(edge_y, dim=-1)
        edge_loss = self.eval_loss(logp_x, p_y)
        return edge_loss, th.mean(edge_pred, dim=-1).squeeze(1)

class BUrumorGCN(th.nn.Module):
    def __init__(self,args):
        super(BUrumorGCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(args.input_features, args.hidden_features)
        self.conv2 = GCNConv(args.input_features + args.hidden_features, args.output_features)
        self.device = args.device
        self.num_features_list = [args.hidden_features * r for r in [1]]
        def creat_network(self, name):
            layer_list = OrderedDict()
            for l in range(len(self.num_features_list)):
                layer_list[name + 'conv{}'.format(l)] = th.nn.Conv1d(
                    in_channels=args.hidden_features,
                    out_channels=args.hidden_features,
                    kernel_size=1,
                    bias=False)
                layer_list[name + 'norm{}'.format(l)] = th.nn.BatchNorm1d(num_features=args.hidden_features)
                layer_list[name + 'relu{}'.format(l)] = th.nn.LeakyReLU()
            layer_list[name + 'conv_out'] = th.nn.Conv1d(in_channels=args.hidden_features,
                                                         out_channels=1,
                                                         kernel_size=1)
            return layer_list
        self.sim_network = th.nn.Sequential(creat_network(self, 'sim_val'))
        mod_self = self
        mod_self.num_features_list = [args.hidden_features]  #
        self.W_mean = th.nn.Sequential(creat_network(mod_self, 'W_mean'))
        self.W_bias = th.nn.Sequential(creat_network(mod_self, 'W_bias'))
        self.B_mean = th.nn.Sequential(creat_network(mod_self, 'B_mean'))
        self.B_bias = th.nn.Sequential(creat_network(mod_self, 'B_bias'))
        self.fc1 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        self.fc2 = th.nn.Linear(args.hidden_features, args.edge_num, bias=False)
        self.dropout = th.nn.Dropout(args.dropout)
        self.eval_loss = th.nn.KLDivLoss(reduction='batchmean')  # mean
        self.bn1 = BatchNorm1d(args.hidden_features + args.input_features)

    def forward(self, data):
        x, edge_index = data.x, data.BU_edge_index
        x1 = copy.copy(x.float())
        x = self.conv1(x, edge_index)
        x2 = copy.copy(x)

        if self.args.edge_infer_bu:
            edge_loss, edge_pred = self.edge_infer(x, edge_index)
        else:
            edge_loss, edge_pred = None, None

        rootindex = data.rootindex
        root_extend = th.zeros(len(data.batch), x1.size(1)).to(self.device)
        batch_size = max(data.batch) + 1
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x1[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_pred)
        x = F.relu(x)
        root_extend = th.zeros(len(data.batch), x2.size(1)).to(self.device)
        for num_batch in range(batch_size):
            index = (th.eq(data.batch, num_batch))
            root_extend[index] = x2[rootindex[num_batch]]
        x = th.cat((x,root_extend), 1)

        x= scatter_mean(x, data.batch, dim=0)
        return x, edge_loss

    def edge_infer(self, x, edge_index):
        row, col = edge_index[0], edge_index[1]
        x_i = x[row - 1].unsqueeze(2)
        x_j = x[col - 1].unsqueeze(1)
        x_ij = th.abs(x_i - x_j)
        sim_val = self.sim_network(x_ij)
        edge_pred = self.fc1(sim_val)
        edge_pred = th.sigmoid(edge_pred)

        w_mean = self.W_mean(x_ij)
        w_bias = self.W_bias(x_ij)
        b_mean = self.B_mean(x_ij)
        b_bias = self.B_bias(x_ij)
        logit_mean = w_mean * sim_val + b_mean
        logit_var = th.log((sim_val ** 2) * th.exp(w_bias) + th.exp(b_bias))

        edge_y = th.normal(logit_mean, logit_var)
        edge_y = th.sigmoid(edge_y)
        edge_y = self.fc2(edge_y)

        logp_x = F.log_softmax(edge_pred, dim=-1)
        p_y = F.softmax(edge_y, dim=-1)
        edge_loss = self.eval_loss(logp_x, p_y)
        return edge_loss, th.mean(edge_pred, dim=-1).squeeze(1)

class EBGCN(th.nn.Module):
    def __init__(self, args):
        super(EBGCN, self).__init__()
        self.args = args
        self.TDrumorGCN = TDrumorGCN(args)
        self.BUrumorGCN = BUrumorGCN(args)
        self.fc = th.nn.Linear((args.hidden_features + args.output_features)*2, args.num_class)

    def forward(self, data):
        TD_x, TD_edge_loss = self.TDrumorGCN(data)
        BU_x, BU_edge_loss = self.BUrumorGCN(data)

        self.x = th.cat((BU_x,TD_x), 1)
        out = self.fc(self.x)
        out = F.log_softmax(out, dim=1)
        return out,  TD_edge_loss, BU_edge_loss
