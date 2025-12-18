# @Author: Yixin Tian
# @File: ml_factor.py
# @Date: 2025/9/8 10:21
# @Software: PyCharm
import os
from operator import index

import feather
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score
from tqdm import tqdm
from tool_tyx.path_data import DataPath
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,StackingRegressor,GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression,LinearRegression

def read_data(date):
    x=feather.read_dataframe(os.path.join(DataPath.train_data_path,f'{date}_train_x.feather'))
    x.replace([np.inf,-np.inf],np.nan,inplace=True)
    y=feather.read_dataframe(os.path.join(DataPath.train_data_path,f'{date}_train_y_ori.feather'))
    y.replace([np.inf,-np.inf],np.nan,inplace=True)
    return [x,y]

def group_mean(true_data:pd.Series,pre_data:np.array,num):
    df = pd.DataFrame(true_data).reset_index(drop=True).merge(pd.DataFrame(pre_data), left_index=True,
                right_index=True).rename(columns={'ret': 'true_value', 0: 'pre_value'})
    df.sort_values(['pre_value'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['group'] = df.index // num
    res = df.groupby('group')['true_value'].mean()
    return res

#决策树/随机森林
def decision_tree_factors(train_data_list,test_data_list):
    # 训练
    train_x,train_y=train_data_list[0].iloc[:,2:],train_data_list[1].ret #train_x,train_x
    # tree_clf=DecisionTreeRegressor(max_depth=70,min_samples_leaf=2,max_features=5,max_leaf_nodes=5)
    tree_clf=RandomForestRegressor(n_estimators=70,min_samples_split=5,max_features=1,max_depth=30,random_state=123)
    # tree_clf=AdaBoostRegressor(DecisionTreeRegressor(max_depth=70,min_samples_leaf=5,max_features=3),
    #                            n_estimators=70)
    # tree_clf=GradientBoostingRegressor(n_estimators=70,max_depth=10,min_samples_split=100)
    # tree_clf=SVR(kernel='linear')

    tree_clf.fit(train_x,train_y)
    pre_train_y=tree_clf.predict(train_x)
    # print(len(pre_train_y))
    res1=group_mean(train_y,pre_train_y,129993).reset_index()
    print('---------------------------------')
    # # 验证
    test_x, test_y = test_data_list[0].iloc[:, 2:], test_data_list[1].ret
    pre_y = tree_clf.predict(test_x)
    res2=group_mean(test_y,pre_y,31132).reset_index()
    print(f'ic on testdata: {np.corrcoef(pre_y, test_y)[0][1]}')
    res=res1.merge(res2, on='group',how='outer').rename(columns={'true_value_x':'train_true','true_value_y':'test_true'})
    print(res)
    res.to_csv(r'C:\Users\admin\Desktop\test.csv',index=False)

def run():
    tar_list = sorted(list(set([x[:8] for x in os.listdir(DataPath.train_data_path)])))
    # test_list = tar_list[-120:]
    # train_list = sorted(list(set(tar_list).difference(set(test_list))))
    train_list=[x for x in tar_list if x>='20220104' and x<='20240201']
    test_list=[x for x in tar_list if x>='20240202' and x<='20240731' and x not in ['20240206', '20240207', '20240208']]
    train_data_list=Parallel(n_jobs=15)(delayed(read_data)(date) for date in tqdm(train_list,desc='train datas are preparing'))
    train_data_list1=pd.concat([x[0] for x in train_data_list])
    train_data_list2=pd.concat([x[1] for x in train_data_list])
    train_data_list=[train_data_list1,train_data_list2]
    test_data_list = Parallel(n_jobs=15)(delayed(read_data)(date) for date in tqdm(test_list,desc='test datas are preparing'))
    test_data_list1 = pd.concat([x[0] for x in test_data_list])
    test_data_list2 = pd.concat([x[1] for x in test_data_list])
    test_data_list=[test_data_list1,test_data_list2]
    decision_tree_factors(train_data_list,test_data_list)

if __name__=='__main__':
    run()