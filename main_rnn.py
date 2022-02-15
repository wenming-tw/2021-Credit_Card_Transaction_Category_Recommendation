import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.rnn_utils import RnnUtils, load_data, log_results
from model import MultiLabelModel

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', dest = 'data_dir', default = '../data/sparse_copy/')
parser.add_argument('--out_dir', dest = 'out_dir', default = '../results/rnn/')
parser.add_argument('--seed', dest = 'random_seed', default = -1, type = int)
parser.add_argument('--test_num', dest = 'test_num', default = 4, type = int)
parser.add_argument('--ws', dest = 'ws', default = 12, type = int)
parser.add_argument('--sparse_feat', dest = 'sparse_feat', default = 1, type = int)
parser.add_argument('--dense_feat', dest = 'dense_feat', default = 1, type = int)
parser.add_argument('--embed_dim', dest = 'embed_dim', default = 2, type = int)
parser.add_argument('--h_dim', dest = 'h_dim', default = 512, type = int)
parser.add_argument('--num_layers', dest = 'num_layers', default = 1, type = int)
parser.add_argument('--pool', dest = 'pool', type = str, default = 'last')
parser.add_argument('--cell', dest = 'cell', type = str, default = 'LSTM')
parser.add_argument('--dropout', dest = 'dropout', default = 0., type = float)
parser.add_argument('--bsize', dest = 'bsize', default = 25000, type = int)
parser.add_argument('--epochs', dest = 'epochs', default = 50, type = int)
parser.add_argument('--gpu', dest = 'gpu', default = 0, type = int)
parser.add_argument('--lr', dest = 'lr', default = 1e-4, type = float)
parser.add_argument('--note', dest = 'note', type = str, default = '')
args = parser.parse_args()
print(vars(args))

X, Y, x_pred, feat_sparse = load_data(args.ws, args.sparse_feat, args.dense_feat, args.data_dir, __file__[:-3])
print('x_seq: {}, dense: {}, sparse: {}'.format(X[0][0].shape, X[0][1].shape, feat_sparse.shape))

if args.random_seed == -1:
    x_train, x_test, y_train, y_test = X[:-args.test_num], X[-args.test_num:], Y[:-args.test_num], Y[-args.test_num:]
else:
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=args.test_num, random_state=args.random_seed)
print('train: {}, test: {}'.format(len(x_train), len(x_test)))

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
pred_label = [2, 6, 10, 12, 13, 15, 18, 19, 21, 22, 25, 26, 36, 37, 39, 48]

args.x_dim = X[0][0].shape[-1]
args.sparse_nums = [feat_sparse[:, i].max().item()+1 for i in range(feat_sparse.shape[1])] if args.sparse_feat else None
args.dense_dim = X[0][1].shape[-1] if args.dense_feat else None
args.out_dim = len(pred_label)

model = MultiLabelModel(args).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.9)
criterion = nn.BCEWithLogitsLoss()
model_utils = RnnUtils(model, args.sparse_feat, args.dense_feat, device)

k = 3
te_metrics = model_utils.evaluate(feat_sparse, x_test, y_test, k, args.bsize)
print('Before training,', ', '.join(['{}: {:.4f}'.format(key, value) for key, value in te_metrics.items()]))

experiment_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S") 
print('experiment_id:', experiment_id)
os.makedirs(args.out_dir + 'model/', exist_ok=True)

his_metrics = list()
best_state, best_metrics, best_epoch = None, {'auc': 0., 'ndcg@'+str(k): 0.}, 0.
pbar = tqdm(range(args.epochs), bar_format='{l_bar}{bar:10}{r_bar}')
for ep in pbar:
    if best_epoch + 10 <= ep: 
        break

    train_loss = 0
    for xs, y in zip(x_train, y_train):
        train_loss += model_utils.learn(feat_sparse, xs, y, optimizer, criterion, args.bsize)
    #scheduler.step()
    
    te_metrics = model_utils.evaluate(feat_sparse, x_test, y_test, k, args.bsize)
    his_metrics.append(te_metrics)   

    if te_metrics['ndcg@'+str(k)] > best_metrics['ndcg@'+str(k)]:
        best_epoch = ep
        best_metrics = te_metrics
        best_state = model.state_dict()
        torch.save(best_state, args.out_dir + 'model/{}.pt'.format(experiment_id))
        pbar.set_description('Epoch: {:03d} | train_loss: {:.4f} | '.format(ep, train_loss) + ', '.join(['{}: {:.4f}'.format(key, value) for key, value in te_metrics.items()]) + ' | ')

    np.random.seed(args.random_seed)
    np.random.shuffle(x_train)
    np.random.seed(args.random_seed)
    np.random.shuffle(y_train)

model.load_state_dict(torch.load(args.out_dir + 'model/{}.pt'.format(experiment_id)))
te_metrics = model_utils.evaluate(feat_sparse, x_test, y_test, k, args.bsize)
print('Epoch: {:03d} | '.format(best_epoch) + ', '.join(['{}: {:.4f}'.format(key, value) for key, value in te_metrics.items()]))
log_results(experiment_id, args, best_epoch, te_metrics)

# predict 
chid2idx = np.load('../data/chid2idx.npy', allow_pickle=True).item()

with torch.no_grad():
    model.eval()
    x = x_pred[0].to(device)
    sparse = feat_sparse.to(device) if args.sparse_feat else None
    dense = x_pred[1].to(device) if args.dense_feat else None

    loader = DataLoader(torch.arange(x.shape[0]), batch_size=args.bsize, shuffle=False)
    y_prob = torch.cat([torch.sigmoid(model(x[batch], sparse[batch], dense[batch])).cpu() for batch in loader], dim=0)    

    top_out = torch.tensor(pred_label)[torch.argsort(y_prob, descending=True, dim=1)]
    df_out = pd.DataFrame(chid2idx.keys(), columns=['chid'])
    df_out[['top1', 'top2', 'top3']] = top_out[:, :3].numpy()
    df_out.to_csv(args.out_dir+'out_{}.csv'.format(experiment_id), index=False)