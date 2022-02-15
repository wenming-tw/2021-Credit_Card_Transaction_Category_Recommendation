import torch
from torch import nn
import torch.nn.functional as F

class MultiLabelModel(nn.Module):
    def __init__(self, args):
        super(MultiLabelModel, self).__init__()
        if args.cell == 'LSTM':
            self.rnn = nn.LSTM(args.x_dim, args.h_dim, num_layers=args.num_layers, batch_first=True)
        elif args.cell == 'GRU':
            self.rnn = nn.GRU(args.x_dim, args.h_dim, num_layers=args.num_layers, batch_first=True)
        elif args.cell == 'RNN':
            self.rnn = nn.RNN(args.x_dim, args.h_dim, num_layers=args.num_layers, batch_first=True)
        self.pool = args.pool

        in_dim = args.h_dim if self.pool != 'concat' else args.h_dim*args.ws
        if args.sparse_feat:
            self.user_embed = nn.Embedding(args.sparse_nums[0], len(args.sparse_nums[1:])*args.embed_dim)
            self.feat_embed = nn.Embedding(sum(args.sparse_nums[1:]), args.embed_dim)
            self.feat_shift = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(torch.tensor(args.sparse_nums[1:-1]), dim=0)])        
            self.userW = nn.Linear(2*len(args.sparse_nums[1:])*args.embed_dim, args.h_dim)
            in_dim += args.h_dim

        if args.dense_feat:
            self.featW = nn.Linear(args.dense_dim, args.h_dim)
            in_dim += args.h_dim

        #self.fusion_layer = nn.Linear(in_dim, in_dim//4)
        #self.out_layer = nn.Linear(in_dim//4, args.out_dim)
        self.fusion_layer = nn.Linear(in_dim, args.out_dim*4)
        self.out_layer = nn.Linear(args.out_dim*4, args.out_dim)        

    def init_parameters(self):
        nn.init.xavier_uniform_(self.user_embed.weight.data)
        nn.init.xavier_uniform_(self.feat_embed.weight.data)
        nn.init.normal_(self.userW.weight.data, 0, 0.005)
        nn.init.normal_(self.featW.weight.data, 0, 0.005)
        nn.init.normal_(self.fusion_layer.weight.data, 0, 0.005)
        nn.init.normal_(self.out_layer.weight.data, 0, 0.005)

    def forward(self, x, u_sparse, dense, ret_h=False):
        h = [self.pooling(self.rnn(x)[0])]
        if u_sparse is not None:
            u_embed = self.user_embed(u_sparse[:,0])
            u_feat = u_sparse[:,1:] + self.feat_shift.to(u_sparse.device)
            f_embed = self.feat_embed(u_feat).reshape(u_feat.shape[0], -1) # (batch, u_dim)
            h += [F.normalize(self.userW(torch.cat([u_embed, f_embed], dim=-1)), p=2, dim=1)]

        if dense is not None:
            dense = self.featW(dense)
            h += [dense]

        h = torch.relu(self.fusion_layer(torch.cat(h, dim=-1)))
        out = self.out_layer(h)

        if ret_h:
            return out, u_embed
        return out

    def pooling(self, h):
        if self.pool == 'last':
            h = h[:, -1]
        elif self.pool == 'max':
            h = torch.max(h, dim=1)[0]
        elif self.pool == 'mean':
            h = torch.mean(h, dim=1)
        elif self.pool == 'sum':
            h = torch.sum(h, dim=1)            
        elif self.pool == 'concat':
            h = h.reshape(h.shape[0], -1)
        return h


