import os
import torch
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from utils.tree_utils import CatUtils, load_data, log_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest = 'data_dir', default = '../data/sparse/')
    parser.add_argument('--out_dir', dest = 'out_dir', default = '../results/cat/')
    parser.add_argument('--feat_dir', dest = 'feat_dir', default = '')
    parser.add_argument('--seed', dest = 'random_seed', default = 2021, type = int)
    parser.add_argument('--test_num', dest = 'test_num', default = 4, type = int)
    parser.add_argument('--ws', dest = 'ws', default = 12, type = int)
    parser.add_argument('--sparse_feat', dest = 'sparse_feat', default = 1, type = int)
    parser.add_argument('--dense_feat', dest = 'dense_feat', default = 1, type = int)
    parser.add_argument('--max_depth', dest = 'max_depth', default = 8, type = int)
    parser.add_argument('--reg_lambda', dest = 'reg_lambda', default = 1., type = float)
    parser.add_argument('--lr', dest = 'learning_rate', default = 0.1, type = float)
    parser.add_argument('--num_boost_round', dest = 'num_boost_round', default = 5000, type = int)
    parser.add_argument('--note', dest = 'note', type = str, default = '')
    args = parser.parse_args()
    print(vars(args))

    params = {
        'iterations': args.num_boost_round,
        'depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'l2_leaf_reg': args.reg_lambda,
        'eval_metric': 'AUC',
        'random_seed': 0,
        'early_stopping_rounds': 100,
        'verbose': False,
        'use_best_model': True,
        'task_type': 'GPU',
        'devices':'1'
    }
    pred_label = [2,  6, 10, 12, 13, 15, 18, 19, 21, 22, 25, 26, 36, 37, 39, 48]

    X, Y, Mask, x_pred, categorical_feature = load_data(args.ws, args.sparse_feat, args.dense_feat, args.data_dir, args.feat_dir, method='lgb', cache='tree')
    catUtils = CatUtils(categorical_feature)

    if args.random_seed == -1:
        X_train, X_test, Y_train, Y_test = X[:-args.test_num], X[-args.test_num:], Y[:-args.test_num], Y[-args.test_num:]
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=args.test_num, random_state=args.random_seed)
    print('train: {}, test: {}'.format(len(X_train), len(X_test)))

    x_train = sparse.vstack(list(X_train))
    y_train = np.concatenate(list(Y_train), axis=0)
    x_test = sparse.vstack(list(X_test))
    y_test = np.concatenate(list(Y_test), axis=0)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    experiment_id = datetime.now().strftime("%m_%d_%Y_%H_%M_%S") 
    print('experiment_id:', experiment_id)
    models, eval_reuslts = catUtils.build_multiLabelModel(x_train, y_train, x_test, y_test, params)

    te_metrics = catUtils.evaluate(models, X_test, Y_test)
    print(', '.join(['{}: {:.4f}'.format(key, value) for key, value in te_metrics.items()]))

    os.makedirs(args.out_dir+'model_{}/'.format(experiment_id), exist_ok=True)
    for i, model in enumerate(models):
        model.save_model(args.out_dir+'model_{}/cat_{:02d}.dump'.format(experiment_id, i))
    log_results(experiment_id, args, te_metrics, eval_reuslts)

    chid2idx = np.load('../data/chid2idx.npy', allow_pickle=True).item()
    y_prob = torch.FloatTensor(catUtils.predict(models, x_pred))
    top_out = torch.tensor(pred_label)[torch.argsort(y_prob, descending=True, dim=1)]

    df_out = pd.DataFrame(chid2idx.keys(), columns=['chid'])
    df_out[['top1', 'top2', 'top3']] = top_out[:, :3].numpy()
    df_out.to_csv(args.out_dir+'out_{}.csv'.format(experiment_id), index=False)
    torch.save(y_prob, args.out_dir+'prob_{}.pt'.format(experiment_id))
