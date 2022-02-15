import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import kurtosis, skew
from sklearn.metrics import ndcg_score
from torch.utils.data import DataLoader
from torchmetrics import functional as F 

to_save = ['random_seed', 'ws', 'embed_dim', 'h_dim', 'num_layers', 'pool', 'cell', 'dropout', 'bsize', 'epochs', 'lr', 'note']
def log_results(experiment_id, args, epoch, metrics, name='logs_final.csv', to_save=to_save):
    dest = args.out_dir + name
    args_dict = vars(args)

    log_vars = {k: [args_dict[k]] for k in to_save if k in args_dict}
    log_vars['experiment_id'] = [experiment_id]
    log_vars['epoch'] = [epoch]
    log_vars = {**log_vars, **{k: [v] for (k, v) in metrics.items()}}
    df = pd.DataFrame.from_dict(log_vars)
    
    if os.path.exists(dest):
        df.to_csv(dest, header=False, index=False, mode='a')
    else:
        df.to_csv(dest, header=True, index=False, mode='w')

def gen_feat_dense(ws, x_seq, top3_seq, feat_seq):
    feat_dense = []
    idx_list = list(range(4, ws+1, 4))
    for i in idx_list:
        scale = x_seq[:, -i:].sum(dim=-1, keepdim=True)
        scale[scale > 0] = 1.
        feat_dense += [x_seq[:, -i:].sum(dim=1) / (scale.sum(dim=1)+1e-9)]

    # top3 personal mean
    scale = top3_seq.sum(dim=-1, keepdim=True)
    scale[scale > 0] = 1.
    feat_dense += [top3_seq.sum(dim=1) / (scale.sum(dim=1)+1e-9)]  
    # top3 global mean
    feat_dense += [top3_seq.reshape(-1, top3_seq.shape[-1]).mean(dim=0).repeat(top3_seq.shape[0], 1)]

    # personal activity
    count = top3_seq[:, -ws:].sum(dim=-1, keepdim=True)
    count[count > 0] = 1.
    feat_dense += [count.mean(dim=1)]

    feat_dense += [x_seq.std(dim=1)]
    feat_dense += [top3_seq.std(dim=1)]
    feat_dense += [torch.tensor(skew(top3_seq, axis=1))]
    feat_dense += [torch.tensor(kurtosis(top3_seq, axis=1))]
    
    feat_dense += [feat_seq[:,-1]]
    feat_dense += [feat_seq.mean(dim=1)]
    feat_dense += [feat_seq.std(dim=1)]
    feat_dense += [torch.tensor(skew(feat_seq, axis=1))]
    feat_dense += [torch.tensor(kurtosis(feat_seq, axis=1))]    

    feat_dense = torch.cat(feat_dense, dim=-1)
    return feat_dense

def load_data(ws, sparse_feat, dense_feat, data_dir='../data/sparse/', cache=''):
    cache_dir = '../data/cache/{}/ws{}_sp{}_de{}/'.format(cache, ws, sparse_feat, dense_feat)
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.isfile(cache_dir+'X.pt') and os.path.isfile(cache_dir+'Y.pt') and os.path.isfile(cache_dir+'x_pred.pt') :
        X = torch.load(cache_dir+'X.pt')
        Y = torch.load(cache_dir+'Y.pt')
        x_pred = torch.load(cache_dir+'x_pred.pt')
        feat_sparse = torch.load(cache_dir+'feat_sparse.pt') if sparse_feat else None

        print('load cache_dir:', cache_dir)
        return X, Y, x_pred, feat_sparse

    x_datas = [torch.load(data_dir+'x_{:02d}.pt'.format(i)).float() for i in range(1, 25)]
    top3_datas = [torch.load(data_dir+'top3_{:02d}.pt'.format(i)).float() for i in range(1, 25)]
    feat_datas = [torch.load(data_dir+'feat_{:02d}.pt'.format(i)).float() for i in range(1, 25)]
    user_sparse = torch.load(data_dir+'user_sparse.pt')
    user_sparse = torch.cat([torch.arange(user_sparse.shape[0], dtype=torch.long).reshape(-1, 1), user_sparse], dim=-1)
    user_dense = torch.load(data_dir+'user_dense.pt')

    X, Y = [], []
    for i in tqdm(range(ws, len(x_datas))):
        x_seq = torch.stack(x_datas[i-ws:i], dim=1).to_dense()
        top3_seq = torch.stack(top3_datas[:i], dim=1).to_dense().round()
        x = [torch.cat([x_seq, top3_seq[:, -ws:]], dim=-1), None]
        y = top3_datas[i].to_dense()

        if dense_feat:
            feat_seq = torch.stack(feat_datas[i-ws:i], dim=1).to_dense()
            x[1] = torch.cat([user_dense, gen_feat_dense(ws, x_seq, top3_seq, feat_seq)], dim=-1)

        X.append(x)
        Y.append(y)

    ## x_pred
    i = len(x_datas)
    x_pred_seq = torch.stack(x_datas[i-ws:i], dim=1).to_dense()
    top3_seq = torch.stack(top3_datas[:i], dim=1).to_dense().round()
    x_pred = [torch.cat([x_seq, top3_seq[:, -ws:]], dim=-1), None]    

    if dense_feat:
        feat_seq = torch.stack(feat_datas[i-ws:i], dim=1).to_dense()
        x_pred[1] = torch.cat([user_dense, gen_feat_dense(ws, x_pred_seq, top3_seq, feat_seq)], dim=-1) 

    assert X[0][0].shape[-1] == x_pred[0].shape[-1]
    if dense_feat:
        assert X[0][1].shape[-1] == x_pred[1].shape[-1]

    torch.save(X, cache_dir+'X.pt')
    torch.save(Y, cache_dir+'Y.pt')
    torch.save(x_pred, cache_dir+'x_pred.pt')
    if sparse_feat:
        torch.save(user_sparse, cache_dir+'feat_sparse.pt')
    print('save cache_dir:', cache_dir)
    
    return X, Y, x_pred, user_sparse

class RnnUtils():
    def __init__(self, model, sparse_feat, dense_feat, device):
        self.model = model
        self.sparse_feat = sparse_feat
        self.dense_feat = dense_feat
        self.device = device
        
    def learn(self, feat_sparse, xs, y, optimizer, criterion, bsize=1):
        self.model.train()        
        mask = y.sum(axis=1) > 0
        
        x = xs[0][mask].to(self.device)
        y = y[mask].to(self.device)
        feat_sparse = feat_sparse[mask].to(self.device) if self.sparse_feat else None
        feat_dense = xs[1][mask].to(self.device) if self.dense_feat else None

        total_loss = 0
        loader = DataLoader(torch.arange(y.shape[0]), batch_size=bsize, shuffle=True)
        for batch in loader:
            optimizer.zero_grad()
            x_batch = x[batch]
            y_batch = y[batch]
            sparse = feat_sparse[batch] if self.sparse_feat else None
            dense = feat_dense[batch] if self.dense_feat else None

            y_prob = self.model(x_batch, sparse, dense)         
            loss = criterion(y_prob, y_batch.round())

            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache() 

            total_loss += loss.item()
        return total_loss
        
    def evaluate(self, feat_sparse, Xs, Ys, k=3, bsize=1):
        auc, recall, precision, ndcg = list(), list(), list(), list()
        self.model.eval()
        with torch.no_grad():
            for xs, y in zip(Xs, Ys):
                mask = y.sum(axis=1) > 0
                x = xs[0][mask].to(self.device)
                y = y[mask].to(self.device)
                sparse = feat_sparse[mask].to(self.device) if self.sparse_feat else None
                dense = xs[1][mask].to(self.device) if self.dense_feat else None
                
                bsize = y.shape[0] if bsize == 0 else bsize
                loader = DataLoader(torch.arange(y.shape[0]), batch_size=bsize, shuffle=False)
                y_prob = torch.cat([torch.sigmoid(self.model(x[batch], sparse[batch], dense[batch])) for batch in loader], dim=0)
                
                auc += [F.auroc(y_prob.ravel(), y.round().ravel().long(), pos_label=1).item()] 
                recall += [F.recall(y_prob, y.round().long(), top_k=k).item()]
                precision += [F.precision(y_prob, y.round().long(), top_k=k).item()] 
                ndcg += [ndcg_score(y.cpu(), y_prob.cpu(), k=k, ignore_ties=True)]
                torch.cuda.empty_cache()

        ret = {
            'auc': np.mean(auc).round(6),
            'recall@'+str(k): np.mean(recall).round(6), 
            'precision@'+str(k): np.mean(precision).round(6),
            'ndcg@'+str(k): np.mean(ndcg).round(6), 
        } 
        return ret


