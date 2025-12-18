# @Author: Yixin Tian
# @File: ml_factor.py
# @Date: 2025/9/8 10:21
# @Software: PyCharm
import os
import feather
import lightgbm
import numpy as np
import pandas as pd
from erfa import TURNAS
from joblib import Parallel,delayed
from tqdm import tqdm
from tool_tyx.path_data import DataPath
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,StackingRegressor,GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from tool_tyx.tyx_funcs import get_ml_data


def group_mean(aa):
    ret = pd.read_excel(r"\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\Exit_4_data_4个时段_orderbook_beta_722.xlsx")
    ret = ret[['TICKER', 'DATE', '平均卖出收益_1100_1400']]
    ret.loc[ret['平均卖出收益_1100_1400'] > 0.4, '平均卖出收益_1100_1400'] = 0.4
    ret['DATE'] = ret['DATE'].astype(str)
    ret = ret[['TICKER', 'DATE', '平均卖出收益_1100_1400']]
    ret['ret'] = ret['平均卖出收益_1100_1400']
    ret = ret[['TICKER', 'DATE', 'ret']]
    ret.loc[ret['ret'] > 0.4, 'ret'] = 0.4

    ret['DATE'] = ret['DATE'].astype(str)
    zz = pd.merge(ret, aa, how='inner', on=['TICKER', 'DATE'])
    usename = list(np.setdiff1d(aa.columns, ['TICKER', 'DATE']))[0]
    # bins = pd.qcut(zz[usename], 10)
    zz['group'] = pd.qcut(zz[usename], 10, labels=False, duplicates='drop') + 1
    res = zz.groupby('group')['ret'].mean().reset_index(drop=False)
    print(res)
    return res

#决策树/随机森林
def decision_tree_factors(train_data_list,valid_data_list,test_data_list):
    # 训练
    train_x, train_y = train_data_list[0], train_data_list[1]  # train_x,train_x
    valid_x, valid_y = valid_data_list[0], valid_data_list[1]
    test_x, test_y = test_data_list[0], test_data_list[1]

    # lightgbm---------------------------------------------
    # params={'task': 'train',
    #         'boosting_type': 'gbdt',
    #         'objective': 'regression_l1',
    #         'max_depth': -1,
    #         'num_leaves': 31,
    #         'lambda_l1': 1.0,
    #         'lambda_l2': 5.0,
    #         'learning_rate': 0.01,
    #         'n_estimators': 300,
    #         'max_bin': 255,
    #         'min_split_gain': 0.05,
    #         'min_data_in_leaf': 50,
    #         'feature_fraction': 0.5,
    #         'bagging_fraction': 0.8,
    #         'bagging_freq': 5,
    #         'verbose': -1}
    # train_data=lightgbm.Dataset(train_x,label=train_y)
    # validation_data=lightgbm.Dataset(valid_x,label=valid_y)
    # callback = [lightgbm.early_stopping(stopping_rounds=20, verbose=True),lightgbm.log_evaluation(period=10, show_stdv=True)]
    # def IC(y_true,train_data):
    #     y_pred=train_data.get_label()
    #     return 'IC',np.corrcoef(y_true,y_pred)[0][1],True
    # tree_clf=lightgbm.train(params,train_data,num_boost_round=100,valid_sets=[validation_data],callbacks=callback,feval=IC)

    # 其他集成学习-------------------------------------------
    # tree_clf=RandomForestRegressor(n_estimators=72,min_samples_split=2,max_features=2,max_depth=2,min_samples_leaf=2,random_state=66)#0.016

    # tree_clf=RandomForestRegressor(n_estimators=72,min_samples_split=2,max_features=2,max_depth=2,min_samples_leaf=2,random_state=777)#0.024
    tree_clf=RandomForestRegressor(n_estimators=72,min_samples_split=2,max_features=2,max_depth=2,min_samples_leaf=2,random_state=99)#0.0257
    # tree_clf=AdaBoostRegressor(DecisionTreeRegressor(max_depth=10,random_state=123), #min_samples_leaf=2,max_features=2,random_state=234
    #                            n_estimators=10,random_state=123) #n_estimators=100
    # tree_clf=GradientBoostingRegressor(n_estimators=10,max_depth=2,criterion='squared_error',subsample=0.5,
    #                                    random_state=123)#min_samples_split=3,max_features='log2'
    # tree_clf=SVR()
    # 模型训练 ==========================================
    tree_clf.fit(train_x, train_y)
    # # 特征重要性
    # importances=tree_clf.feature_importances_
    # feathure_names=train_x.columns.tolist()
    # importance_dict=dict(zip(feathure_names,importances))
    # for name,im in sorted(importance_dict.items(),key=lambda item:item[1],reverse=True):
    #     print(f'{name}:{importances}')
    # 训练数据-----------------------------------------------------

    # pre_train_y=tree_clf.predict(train_x)
    # pre_train_y = pd.DataFrame(pre_train_y, index=train_y.index, columns=['pred_train'])  # .reset_index()
    # feather.write_dataframe(pre_train_y,r'C:\Users\admin\Desktop\pred_train.feather')
    # print('train:')
    # group_mean(pre_train_y)
    # print('------------------------')
    # 验证数据-----------------------------------------------------
    pre_valid_y = tree_clf.predict(valid_x)
    # print(f'ic on testdata: {np.corrcoef(pre_valid_y, valid_y)[0][1]}')
    pre_valid_y = pd.DataFrame(pre_valid_y, index=valid_y.index, columns=['pred_valid'])  # .reset_index()
    # feather.write_dataframe(pre_valid_y, r'C:\Users\admin\Desktop\pred_valid.feather')
    print('valid:')
    group_mean(pre_valid_y)
    # print('------------------------')
    # 测试数据-----------------------------------------------------
    # pre_test_y = tree_clf.predict(test_x)
    # pre_test_y = pd.DataFrame(pre_test_y, index=test_y.index, columns=['pred_test']).reset_index()
    # # feather.write_dataframe(pre_test_y, r'C:\Users\admin\Desktop\pred_test.feather')
    # # print(pre_test_y)
    # print('test:')
    # group_mean(pre_test_y)


def run(path):
    train_data_list,valid_data_list,test_data_list=get_ml_data(path)
    decision_tree_factors(train_data_list,valid_data_list,test_data_list)

if __name__=='__main__':
    run(path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\rolling_generation_202508_sz_auction_end_20240815_910_big_order\train')
    # run(path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\rolling_generation_202508_sz_auction_end_20240815_909_big_order\train')

