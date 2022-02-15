import os
import torch
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from tqdm import tqdm
from scipy import sparse
from scipy.stats import kurtosis, skew
from sklearn.metrics import ndcg_score
from torchmetrics import functional as F
from catboost import CatBoostClassifier

def gen_feat_dense(ws, x_seq, top3_seq, feat_seq, other_seq):
    feat_dense = []
    # season
    index_list = list(range(0, ws, 3)) + [ws]
    for i, j in zip(index_list[:-1], index_list[1:]):
        feat_dense += [x_seq[:,i:j].mean(dim=1)]
        feat_dense += [x_seq[:,i:j].std(dim=1)]

    x_pct = x_seq / (x_seq.sum(-1, keepdim=True)+1e-9) # 500000, ws, 16
    feat_dense += [x_pct[:,-1]]
    feat_dense += [x_pct.mean(dim=1)]
    feat_dense += [x_pct.std(dim=1)]

    # nan mean
    scale = top3_seq.sum(dim=-1, keepdim=True)
    scale[scale > 0] = 1.
    feat_dense += [x_seq.sum(dim=1) / (scale.sum(dim=1)+1e-9)]
    feat_dense += [top3_seq.sum(dim=1) / (scale.sum(dim=1)+1e-9)]
    # global mean
    feat_dense += [x_seq.reshape(-1, x_seq.shape[-1]).mean(dim=0).repeat(x_seq.shape[0], 1)]
    feat_dense += [top3_seq.reshape(-1, top3_seq.shape[-1]).mean(dim=0).repeat(top3_seq.shape[0], 1)]
    # x_seq stat
    feat_dense += [x_seq.mean(dim=1)]
    feat_dense += [x_seq.std(dim=1)]
    # top3_seq stat
    feat_dense += [top3_seq.mean(dim=1)]
    feat_dense += [top3_seq.std(dim=1)]
    feat_dense += [torch.tensor(skew(top3_seq, axis=1))]
    feat_dense += [torch.tensor(kurtosis(top3_seq, axis=1))]

    feat_dense += [top3_seq[:,-3:].mean(dim=1)]
    feat_dense += [top3_seq[:,-3:].std(dim=1)]
    feat_dense += [torch.tensor(skew(top3_seq[:,-3:], axis=1))]
    feat_dense += [torch.tensor(kurtosis(top3_seq[:,-3:], axis=1))]    

    # feat_seq : 500000, ws , 16, (1 + 4 + 4 + 15 + 15 + 1)
    feat_dense += [feat_seq[:,-1, :, :].reshape(feat_seq.shape[0], -1)]
    # txn_cnt
    txn_cnt_seq = feat_seq[:,:,:,0].reshape(feat_seq.shape[0], ws, -1) # 500000, ws, 16
    feat_dense += [txn_cnt_seq.mean(dim=1)] # 500000, 16
    feat_dense += [txn_cnt_seq.std(dim=1)] # 500000, 16
    feat_dense += [txn_cnt_seq[:,-3:].mean(dim=1)] # 500000, 16
    feat_dense += [txn_cnt_seq[:,-3:].std(dim=1)] # 500000, 16    

    # cnt
    cnt_seq = feat_seq[:,:,:,1:5].reshape(feat_seq.shape[0], ws, 16, -1) # 500000, ws, 16, 4
    cnt_pct = cnt_seq.sum(2) / (cnt_seq.sum(-1).sum(-1, keepdim=True)+1e-9) # 500000, ws, 4
    feat_dense += [cnt_pct[:,-1]]
    feat_dense += [cnt_pct.mean(dim=1)]
    feat_dense += [cnt_pct.std(dim=1)]
    feat_dense += [cnt_pct[:,-3:].mean(dim=1)]
    feat_dense += [cnt_pct[:,-3:].std(dim=1)]    

    feat_dense += [cnt_seq.mean(dim=1).reshape(cnt_seq.shape[0], -1)]
    feat_dense += [cnt_seq.std(dim=1).reshape(cnt_seq.shape[0], -1)]
    feat_dense += [cnt_seq[:,-3:].mean(dim=1).reshape(cnt_seq.shape[0], -1)]
    feat_dense += [cnt_seq[:,-3:].std(dim=1).reshape(cnt_seq.shape[0], -1)]    
    # pct
    pct_seq = feat_seq[:,:,:,5:9].reshape(feat_seq.shape[0], ws, 16, -1) # 500000, ws, 16, 4
    pct_pct = pct_seq.sum(2) / (pct_seq.sum(-1).sum(-1, keepdim=True)+1e-9) # 500000, ws, 4
    feat_dense += [pct_pct[:,-1]]
    feat_dense += [pct_pct.mean(dim=1)]
    feat_dense += [pct_pct.std(dim=1)]
    feat_dense += [pct_pct[:,-3:].mean(dim=1)]
    feat_dense += [pct_pct[:,-3:].std(dim=1)]    
 
    # card cnt
    cnt_seq = feat_seq[:,:,:,9:24].reshape(feat_seq.shape[0], ws, 16, -1) # 500000, ws, 16, 15
    cnt_pct = cnt_seq.sum(2) / (cnt_seq.sum(-1).sum(-1, keepdim=True)+1e-9) # 500000, ws, 15
    feat_dense += [cnt_pct[:,-1]]
    feat_dense += [cnt_pct.mean(dim=1)]
    feat_dense += [cnt_pct.std(dim=1)]
    feat_dense += [cnt_pct[:,-3:].mean(dim=1)]
    feat_dense += [cnt_pct[:,-3:].std(dim=1)]    

    # card pct
    pct_seq = feat_seq[:,:,:,24:].reshape(feat_seq.shape[0], ws, 16, -1) # 500000, ws, 16, 15
    pct_pct = pct_seq.sum(2) / (pct_seq.sum(-1).sum(-1, keepdim=True)+1e-9) # 500000, ws, 15
    feat_dense += [pct_pct[:,-1]]
    feat_dense += [pct_pct.mean(dim=1)]
    feat_dense += [pct_pct.std(dim=1)]    
    feat_dense += [pct_pct[:,-3:].mean(dim=1)]
    feat_dense += [pct_pct[:,-3:].std(dim=1)]       
    
    # personal activity
    count = top3_seq[:, -ws:].sum(dim=-1, keepdim=True)
    count[count > 0] = 1.
    feat_dense += [count.mean(dim=1)]
    feat_dense += [count[:,-3:].mean(dim=1)]

    # other_seq stat 500000, ws, 33
    feat_dense += [other_seq[:,-1]]
    feat_dense += [other_seq.mean(dim=1)]
    feat_dense += [other_seq.std(dim=1)]
    feat_dense += [other_seq[:,-3:].mean(dim=1)]
    feat_dense += [other_seq[:,-3:].std(dim=1)]    

    return torch.cat(feat_dense, dim=-1)

def load_data(ws, sparse_feat, dense_feat, data_dir='../data/sparse/', feat_dir='', method='xgb', cache=''):
    cache_dir = '../data/cache/{}/ws{}_sp{}_de{}{}/'.format(cache, ws, sparse_feat, dense_feat, feat_dir[-10:-1])
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.isfile(cache_dir+'X.pt') and os.path.isfile(cache_dir+'Y.pt') and os.path.isfile(cache_dir+'Mask.pt') and os.path.isfile(cache_dir+'x_pred.pt') and os.path.isfile(cache_dir+'feature_types.pt'):
        print('load cache_dir:', cache_dir)
        X = torch.load(cache_dir+'X.pt')
        Y = torch.load(cache_dir+'Y.pt')
        Mask = torch.load(cache_dir+'Mask.pt')
        x_pred = torch.load(cache_dir+'x_pred.pt')
        feature_types = torch.load(cache_dir+'feature_types.pt')
        return X, Y, Mask, x_pred, feature_types[method]
    # load data
    print('load data.')
    x_datas = [torch.load(data_dir+'x_{:02d}.pt'.format(i)).float() for i in range(1, 25)]
    top3_datas = [torch.load(data_dir+'top3_{:02d}.pt'.format(i)).float() for i in range(1, 25)]
    feat_datas = [torch.load(data_dir+'feat_{:02d}.pt'.format(i)).float() for i in range(1, 25)]
    other_datas = [torch.load(data_dir+'other_{:02d}.pt'.format(i)).float() for i in range(1, 25)]
    user_sparse = torch.load(data_dir+'user_sparse.pt')
    user_dense = torch.load(data_dir+'user_dense.pt')

    if feat_dir != '':
        featHs = torch.load(feat_dir+'featHs.pt')

    X, Y, Mask = [], [], []
    for i in tqdm(range(ws, len(x_datas))):
        x_seq = torch.stack(x_datas[i-ws:i], dim=1).to_dense()
        x = x_seq.reshape(x_seq.shape[0], -1)
        y = top3_datas[i].to_dense().numpy()

        if feat_dir != '':
            x = torch.cat([featHs[i-ws], x], dim=-1)
        if dense_feat:
            top3_seq = torch.stack(top3_datas[:i], dim=1).to_dense().round()
            feat_seq = torch.stack(feat_datas[i-ws:i], dim=1).to_dense() # 500000, ws, , 16, (1 + 4 + 4)
            other_seq = torch.stack(other_datas[i-ws:i], dim=1).to_dense()
            feat_dense = gen_feat_dense(ws, x_seq, top3_seq, feat_seq, other_seq)
            x = torch.cat([feat_dense, user_dense, x], dim=-1)
        if sparse_feat:
            x = torch.cat([user_sparse, x], dim=-1)

        mask = y.sum(axis=1) > 0
        X.append(sparse.csr_matrix(x.numpy())[mask])
        Y.append(y[mask])    
        Mask.append(mask)

    if sparse_feat:
        feature_types = {
            'xgb': ['c']*user_sparse.shape[1] + ['q']*(X[0].shape[1]-user_sparse.shape[1]),
            'lgb': list(range(user_sparse.shape[1])), 
            'cat': list(range(user_sparse.shape[1])), 
        }
    else:
        feature_types = {
            'xgb': ['q']*X[0].shape[1],
            'lgb': 'auto',
            'cat': []
        }

    ## x_pred
    i = len(x_datas)
    x_pred_seq = torch.stack(x_datas[i-ws:i], dim=1).to_dense()
    x_pred = x_pred_seq.reshape(x_pred_seq.shape[0], -1)    

    if feat_dir != '':
        predH = torch.load(feat_dir+'predH.pt')
        x_pred = torch.cat([predH, x_pred], dim=-1)
    if dense_feat:
        top3_seq = torch.stack(top3_datas[:i], dim=1).to_dense().round()
        feat_seq = torch.stack(feat_datas[i-ws:i], dim=1).to_dense()
        other_seq = torch.stack(other_datas[i-ws:i], dim=1).to_dense()
        feat_dense = gen_feat_dense(ws, x_pred_seq, top3_seq, feat_seq, other_seq)
        x_pred = torch.cat([feat_dense, user_dense, x_pred], dim=-1)
    if sparse_feat:
        x_pred = torch.cat([user_sparse, x_pred], dim=-1)

    x_pred = sparse.csr_matrix(x_pred.numpy())
    assert X[0].shape[-1] == x_pred.shape[-1]

    torch.save(X, cache_dir+'X.pt')
    torch.save(Y, cache_dir+'Y.pt')
    torch.save(Mask, cache_dir+'Mask.pt')
    torch.save(x_pred, cache_dir+'x_pred.pt')
    torch.save(feature_types, cache_dir+'feature_types.pt')
    print('save cache_dir:', cache_dir)
    
    return X, Y, Mask, x_pred, feature_types[method]

class XgbUtils():
    def __init__(self, enable_categorical, feature_types, ws):
        self.enable_categorical = enable_categorical
        self.feature_types = feature_types
        self.ws = ws

    def build_multiLabelModel(self, x_train, y_train, x_test, y_test, params, num_boost_round):
        models, results = [], []
        for i in range(y_train.shape[1]):
            dtrain = xgb.DMatrix(x_train, label=y_train[:, i].round(), feature_types=self.feature_types, enable_categorical=self.enable_categorical)
            dtest = xgb.DMatrix(x_test, label=y_test[:, i].round(), feature_types=self.feature_types, enable_categorical=self.enable_categorical)        
            print('train model_{}'.format(i))
            evals_result = dict()
            model = xgb.train(params=params, 
                            dtrain=dtrain, 
                            num_boost_round=num_boost_round, 
                            early_stopping_rounds=50, 
                            evals=[(dtest,'test')], 
                            evals_result = evals_result, 
                            verbose_eval=0)                          
            models.append(model)
            results.append(evals_result)
            print('[{}]\ttest-auc: {}\n'.format(model.best_iteration, evals_result['test']['auc'][model.best_iteration]))
        return models, results

    def predict(self, models, x):
        y_prob = []
        data = xgb.DMatrix(data=x, feature_types=self.feature_types, enable_categorical=self.enable_categorical)
        for model in models:
            y_prob.append(model.predict(data, iteration_range=(0, model.best_iteration)).reshape(-1, 1))   
        return np.concatenate(y_prob, axis=1)

    def predict_leaf(self, model, x):
        data = xgb.DMatrix(data=x, feature_types=self.feature_types, enable_categorical=self.enable_categorical)
        y_prob = model.predict(data, iteration_range=(0, model.best_iteration), pred_leaf=True)
        return y_prob

    def evaluate(self, models, Xs, Ys, k=3):
        auc, recall, precision, ndcg = list(), list(), list(), list()
        for x, y_top in zip(Xs, Ys):
            y_prob = self.predict(models, x)
            y_top = torch.FloatTensor(y_top).cuda()
            y_prob = torch.FloatTensor(y_prob).cuda()

            auc += [F.auroc(y_prob.ravel(), y_top.round().ravel().long(), pos_label=1).item()] 
            recall += [F.recall(y_prob, y_top.round().long(), top_k=k).item()]
            precision += [F.precision(y_prob, y_top.round().long(), top_k=k).item()] 
            
            y_top, y_prob = y_top.cpu(), y_prob.cpu()
            ndcg += [ndcg_score(y_top, y_prob, k=k, ignore_ties=True)]        

        ret = {
            'auc': np.mean(auc).round(6),
            'recall@'+str(k): np.mean(recall).round(6), 
            'precision@'+str(k): np.mean(precision).round(6),
            'ndcg@'+str(k): np.mean(ndcg).round(6), 
        } 
        return ret        

class LgbUtils():
    def __init__(self, categorical_feature):
        self.categorical_feature = categorical_feature

    def build_multiLabelModel(self, x_train, y_train, x_test, y_test, params, num_boost_round):
        models, results = [], []
        for i in range(y_train.shape[1]):
            dtrain = lgb.Dataset(x_train, label=y_train[:, i].round())
            dtest = lgb.Dataset(x_test, label=y_test[:, i].round())        
            print('train model_{}'.format(i))
            evals_result = dict()
            model = lgb.train(params, 
                            dtrain, 
                            num_boost_round=num_boost_round, 
                            valid_sets=dtest, 
                            categorical_feature=self.categorical_feature, 
                            callbacks=[lgb.early_stopping(25), lgb.record_evaluation(evals_result)])  
            models.append(model)
            results.append(evals_result)
            print()
        return models, results

    def predict(self, models, x):
        y_prob = []
        for model in models:
            y_prob.append(model.predict(x).reshape(-1, 1))   
        return np.concatenate(y_prob, axis=1)

    def evaluate(self, models, Xs, Ys, k=3):
        auc, recall, precision, ndcg = list(), list(), list(), list()
        for x, y_top in zip(Xs, Ys):
            y_prob = self.predict(models, x)
            y_top = torch.FloatTensor(y_top)
            y_prob = torch.FloatTensor(y_prob)

            auc += [F.auroc(y_prob.ravel(), y_top.round().ravel().long(), pos_label=1).item()] 
            recall += [F.recall(y_prob, y_top.round().long(), top_k=k)]
            precision += [F.precision(y_prob, y_top.round().long(), top_k=k)] 
            ndcg += [ndcg_score(y_top, y_prob, k=k, ignore_ties=True)]        

        ret = {
            'auc': np.mean(auc).round(6),
            'recall@'+str(k): np.mean(recall).round(6), 
            'precision@'+str(k): np.mean(precision).round(6),
            'ndcg@'+str(k): np.mean(ndcg).round(6), 
        } 
        return ret        

class CatUtils():
    def __init__(self, categorical_feature):
        self.categorical_feature = categorical_feature

    def build_multiLabelModel(self, x_train, y_train, x_test, y_test, params):
        models, results = [], []
        x_train = pd.DataFrame.sparse.from_spmatrix(x_train)
        x_test = pd.DataFrame.sparse.from_spmatrix(x_test)
        x_train.iloc[:,self.categorical_feature] = x_train.iloc[:,self.categorical_feature].astype(pd.SparseDtype(int, fill_value=0))
        x_test.iloc[:,self.categorical_feature] = x_test.iloc[:,self.categorical_feature].astype(pd.SparseDtype(int, fill_value=0))        
        for i in range(y_train.shape[1]):
            print('train model_{}'.format(i))
            model = CatBoostClassifier(**params)            
            model.fit(x_train, y_train[:, i].round(), eval_set=(x_test, y_test[:, i].round()), cat_features=self.categorical_feature)
            models.append(model)
            
            evals_result = round(model.get_best_score().get('validation').get('AUC'), 6)
            results.append(evals_result)
            print('[{}]\ttest-auc: {}\n'.format(model.get_best_iteration(), evals_result))
            print()
        return models, results

    def predict(self, models, x):
        x = pd.DataFrame.sparse.from_spmatrix(x)
        x.iloc[:,self.categorical_feature] = x.iloc[:,self.categorical_feature].astype(pd.SparseDtype(int, fill_value=0))
        
        y_prob = []
        for model in models:
            y_prob.append(model.predict_proba(x)[:,1].reshape(-1, 1))   
        return np.concatenate(y_prob, axis=1)

    def evaluate(self, models, Xs, Ys, k=3):
        auc, recall, precision, ndcg = list(), list(), list(), list()
        for x, y_top in zip(Xs, Ys):
            y_prob = self.predict(models, x)
            y_top = torch.FloatTensor(y_top)
            y_prob = torch.FloatTensor(y_prob)

            auc += [F.auroc(y_prob.ravel(), y_top.round().ravel().long(), pos_label=1).item()] 
            recall += [F.recall(y_prob, y_top.round().long(), top_k=k)]
            precision += [F.precision(y_prob, y_top.round().long(), top_k=k)] 
            ndcg += [ndcg_score(y_top, y_prob, k=k, ignore_ties=True)]        

        ret = {
            'auc': np.mean(auc).round(6),
            'recall@'+str(k): np.mean(recall).round(6), 
            'precision@'+str(k): np.mean(precision).round(6),
            'ndcg@'+str(k): np.mean(ndcg).round(6), 
        } 
        return ret        

to_save_lgb = ['feat_dir', 'random_seed', 'sample_seed', 'ws', 'test_num', 'sparse_feat', 'dense_feat', 'max_depth', 'num_leaves', 
                'reg_alpha', 'reg_lambda', 'subsample', 'colsample_bytree', 'learning_rate', 'num_boost_round', 'note']
def log_results(experiment_id, args, metrics, eval_reuslt, name='logs_final.csv', to_save=to_save_lgb):
    dest = args.out_dir + name
    args_dict = vars(args)
    
    log_vars = {k: [args_dict[k]] for k in to_save if k in args_dict}
    log_vars['experiment_id'] = [experiment_id]
    log_vars = {**log_vars, **{k: [v] for (k, v) in metrics.items()}}
    
    for i in range(len(eval_reuslt)):
        log_vars['auc_'+str(i)] = eval_reuslt[i]
    df = pd.DataFrame.from_dict(log_vars)

    if os.path.exists(dest):
        df.to_csv(dest, header=False, index=False, mode='a')
    else:
        df.to_csv(dest, header=True, index=False, mode='w')