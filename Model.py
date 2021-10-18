import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import numpy as np

class CFG_lgbm :
  SEED = 42
  n_splits = 5
  lgb_params = {'boosting_type': 'gbdt','objective': 'multiclass',  'metric': 'multi_logloss',    'num_class':3,
                'n_estimators': 10000,'sub_sample' : 0.7,'colsample_bytree' : 0.6,
                'seed': SEED,'silent':False,'early_stopping_rounds': 100,}
  remove_features = ['res','season','matchesPlayed','homeDraws','result','week','Goal_conceded_away','Goal_conceded_away','homeTeam','awayTeam','matchDate','league']
  TARGET_COL = 'result'

def features_utils_lgbm (train):
  features_columns = [col for col in train.columns if col not in CFG_lgbm.remove_features]
  return features_columns

def divide_train (train,features_columns_lgbm):
  skf = StratifiedKFold(n_splits=CFG_lgbm.n_splits,shuffle=True, random_state=CFG_lgbm.SEED)
  X , y   = train[features_columns_lgbm] , train[CFG_lgbm.TARGET_COL]
  return X,y,skf


def StratifiedKFold_Train(X,y,skf,train,test,features_columns_lgbm) :
    # oof_lgb = np.zeros((train.shape[0],))
    test['target'] = 0
    # lgb_preds = []
    models = []
    for fold_, (trn_idx, val_idx) in enumerate(skf.split(X, train.res)):
        print(50 * '-')
        print('Fold:', fold_ + 1)

        tr_x, tr_y = X.iloc[trn_idx, :], y[trn_idx]
        vl_x, vl_y = X.iloc[val_idx, :], y[val_idx]

        train_data = lgb.Dataset(tr_x, label=tr_y)
        valid_data = lgb.Dataset(vl_x, label=vl_y)

        estimator = lgb.train(CFG_lgbm.lgb_params, train_data, valid_sets=[train_data, valid_data], verbose_eval=100)

        # y_pred_val = estimator.predict(vl_x,num_iteration=estimator.best_iteration)
        # oof_lgb[val_idx] = y_pred_val
        #
        # y_pred_test = estimator.predict(test[features_columns_lgbm], num_iteration=estimator.best_iteration)
        # lgb_preds.append(y_pred_test)
        models.append(estimator)
        print(50*'-')

    return models

def predict (models,features_columns_lgbm,test) :
    lgb_preds = []
    for model in models:
        y_pred_test = model.predict(test[features_columns_lgbm], num_iteration=model.best_iteration)
        lgb_preds.append(y_pred_test)
    lightgbm_preds = np.mean(lgb_preds, axis=0)
    lgb_pred = [np.where(x == max(x))[0][0] for x in lightgbm_preds]
    return  lgb_pred

