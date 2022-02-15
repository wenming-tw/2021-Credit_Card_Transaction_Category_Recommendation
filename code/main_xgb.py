import os
import torch
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from datetime import datetime
from sklearn.model_selection import train_test_split
from utils.tree_utils import XgbUtils, load_data, log_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest = 'data_dir', default = '../data/sparse/')
    parser.add_argument('--out_dir', dest = 'out_dir', default = '../results/xgb/')
    parser.add_argument('--feat_dir', dest = 'feat_dir', default = '')
    parser.add_argument('--seed', dest = 'random_seed', default = -1, type = int)
    parser.add_argument('--test_num', dest = 'test_num', default = 4, type = int)
    parser.add_argument('--ws', dest = 'ws', default = 12, type = int)
    parser.add_argument('--sparse_feat', dest = 'sparse_feat', default = 1, type = int)
    parser.add_argument('--dense_feat', dest = 'dense_feat', default = 1, type = int)
    parser.add_argument('--max_depth', dest = 'max_depth', default = 6, type = int)
    parser.add_argument('--gamma', dest = 'gamma', default = 0, type = float)
    parser.add_argument('--reg_alpha', dest = 'reg_alpha', default = 100, type = float)
    parser.add_argument('--reg_lambda', dest = 'reg_lambda', default = 1., type = float)
    parser.add_argument('--subsample', dest = 'subsample', default = 1., type = float)
    parser.add_argument('--colsample_bytree', dest = 'colsample_bytree', default = 0.5, type = float)
    parser.add_argument('--lr', dest = 'learning_rate', default = 0.05, type = float)
    parser.add_argument('--num_boost_round', dest = 'num_boost_round', default = 10000, type = int)
    parser.add_argument('--gpu_id', dest = 'gpu_id', default = 0, type = int)
    parser.add_argument('--note', dest = 'note', type = str, default = '')
    args = parser.parse_args()
    print(vars(args))

    params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'learning_rate': args.learning_rate, 'lambda': args.reg_lambda, 'alpha': args.reg_alpha, 
            'max_depth': args.max_depth, 'sampling_method':'uniform', 'subsample': args.subsample, 'colsample_bytree': args.colsample_bytree, 'gamma': args.gamma,
            'tree_method': 'gpu_hist', 'predictor': 'gpu_predictor', 'gpu_id': args.gpu_id}
    pred_label = [2,  6, 10, 12, 13, 15, 18, 19, 21, 22, 25, 26, 36, 37, 39, 48]

    enable_categorical = bool(args.sparse_feat)
    X, Y, Mask, x_pred, feature_types = load_data(args.ws, args.sparse_feat, args.dense_feat, args.data_dir, args.feat_dir, method='xgb', cache='tree')
    xgbUtils = XgbUtils(enable_categorical, feature_types, args.ws)

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
    models, eval_reuslts = xgbUtils.build_multiLabelModel(x_train, y_train, x_test, y_test, params, args.num_boost_round)

    te_metrics = xgbUtils.evaluate(models, X_test, Y_test)
    eval_reuslt = [round(eval_reuslts[i]['test']['auc'][models[i].best_iteration], 6) for i in range(len(pred_label))]
    print(', '.join(['{}: {:.6f}'.format(key, value) for key, value in te_metrics.items()]))

    os.makedirs(args.out_dir+'model_{}/'.format(experiment_id), exist_ok=True)
    for i, model in enumerate(models):
        model.save_model(args.out_dir+'model_{}/xgb_{:02d}.json'.format(experiment_id, i))
    log_results(experiment_id, args, te_metrics, eval_reuslt)

    chid2idx = np.load('../data/chid2idx.npy', allow_pickle=True).item()
    y_prob = torch.FloatTensor(xgbUtils.predict(models, x_pred))
    top_out = torch.tensor(pred_label)[torch.argsort(y_prob, descending=True, dim=1)]

    df_out = pd.DataFrame(chid2idx.keys(), columns=['chid'])
    df_out[['top1', 'top2', 'top3']] = top_out[:, :3].numpy()
    df_out.to_csv(args.out_dir+'out_{}.csv'.format(experiment_id), index=False)
    torch.save(y_prob, args.out_dir+'prob_{}.pt'.format(experiment_id))