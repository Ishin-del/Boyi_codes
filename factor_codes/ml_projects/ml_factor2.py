# @Author: Yixin Tian
# @File: ml_factor.py
# @Date: 2025/9/8 10:21
# @Software: PyCharm
import os
import warnings
import feather
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from tqdm import tqdm
from tool_tyx.path_data import DataPath
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,StackingRegressor,GradientBoostingRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
def read_data(date):
    x=feather.read_dataframe(os.path.join(DataPath.train_big_order_path,f'{date}_train_x.feather'))
    x.replace([np.inf,-np.inf],np.nan,inplace=True)
    y=feather.read_dataframe(os.path.join(DataPath.train_big_order_path,f'{date}_train_y_ori.feather'))
    y.replace([np.inf,-np.inf],np.nan,inplace=True)
    return [x,y]

def group_mean(true_data,pre_data,num,label='train'):
    # print(true_data)
    # print('-----------')
    # print(pre_data)
    df = true_data.merge(pre_data, left_index=True,right_index=True)#.rename(columns={'ret': 'true_value', 0: 'pre_value'})
    df.sort_values([f'pred_{label}'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['group'] = df.index // num
    res = df.groupby('group')['ret'].mean()
    return res

#决策树/随机森林
def decision_tree_factors(train_data_list,valid_data_list,test_data_list):
    # 训练
    train_x,train_y=train_data_list[0],train_data_list[1]#train_x,train_x
    valid_x, valid_y = valid_data_list[0], valid_data_list[1]
    test_x,test_y=test_data_list[0],test_data_list[1]

    # tree_clf=DecisionTreeRegressor(max_depth=70,min_samples_leaf=2,max_features=5,max_leaf_nodes=5)
    # tree_clf=RandomForestRegressor(n_estimators=100,min_samples_split=5,max_features=1,max_depth=30,min_samples_leaf=5,random_state=123)
    # tree_clf=AdaBoostRegressor(DecisionTreeRegressor(max_depth=100,min_samples_leaf=5,max_features=3),n_estimators=70) #0.05
    tree_clf=AdaBoostRegressor(DecisionTreeRegressor(max_depth=100,
                                                     #min_samples_leaf=5,
                                                     #max_features=3,
                                                     #min_samples_split=2,
                                                     random_state=234),
                                                     n_estimators=100,
                                                     random_state=123) #0.054
    # tree_clf=GradientBoostingRegressor(n_estimators=70,max_depth=70,min_samples_split=3,subsample=0.9,criterion='squared_error',
    #                                    max_features='log2',random_state=123)

    tree_clf.fit(train_x,train_y)
    # 特征重要性
    # importances=tree_clf.feature_importances_
    # feathure_names=train_x.columns.tolist()
    # importance_dict=dict(zip(feathure_names,importances))
    # for name,im in sorted(importance_dict.items(),key=lambda item:item[1],reverse=True):
    #     print(f'{name}:{importances}')
    # ---------------
    pre_train_y=tree_clf.predict(train_x)
    pre_train_y=pd.DataFrame(pre_train_y,index=train_y.index,columns=['pred_train'])#.reset_index()
    # feather.write_dataframe(pre_train_y,r'C:\Users\admin\Desktop\pred_train.feather')
    # print(pre_train_y)
    # res1=group_mean(train_y,pre_train_y,len(pre_train_y)//10+1).reset_index()
    # 验证

    pre_valid_y = tree_clf.predict(valid_x)
    pre_valid_y=pd.DataFrame(pre_valid_y,index=valid_y.index,columns=['pred_valid'])#.reset_index()
    # feather.write_dataframe(pre_valid_y, r'C:\Users\admin\Desktop\pred_valid.feather')
    # print(pre_valid_y)

    res2=group_mean(valid_y,pre_valid_y,len(pre_valid_y)//10+1,label='valid').reset_index()
    print(res2)
    print(f'ic on testdata: {np.corrcoef(pre_valid_y, valid_y)[0][1]}')
    # res=res1.merge(res2, on='group',how='outer').rename(columns={'true_value_x':'train_true','true_value_y':'test_true'})
    # print(res)
    # res.to_csv(r'C:\Users\admin\Desktop\test.csv',index=False)
    pre_test_y=tree_clf.predict(test_x)
    pre_test_y=pd.DataFrame(pre_test_y,index=test_y.index,columns=['pred_test']).reset_index()
    # feather.write_dataframe(pre_test_y, r'C:\Users\admin\Desktop\pred_test.feather')
    print(pre_test_y)

def run():
    warnings.filterwarnings('ignore')
    tar_list = sorted(list(set([x[:8] for x in os.listdir(DataPath.train_big_order_path)])))
    train_list=[x for x in tar_list if x>='20220104' and x<='20240201']
    valid_list=[x for x in tar_list if x>='20240202' and x<='20240731' and x not in ['20240206', '20240207', '20240208']]
    test_list=[x for x in tar_list if x not in train_list and x not in valid_list]
    train_data_list=Parallel(n_jobs=15)(delayed(read_data)(date) for date in tqdm(train_list,desc='train datas are preparing'))
    train_data_list1=pd.concat([x[0] for x in train_data_list]).set_index(['TICKER','DATE'])
    train_data_list2=pd.concat([x[1] for x in train_data_list]).set_index(['TICKER','DATE'])
    train_data_list=[train_data_list1,train_data_list2]
    valid_data_list = Parallel(n_jobs=15)(delayed(read_data)(date) for date in tqdm(valid_list,desc='valid datas are preparing'))
    valid_data_list1 = pd.concat([x[0] for x in valid_data_list]).set_index(['TICKER','DATE'])
    valid_data_list2 = pd.concat([x[1] for x in valid_data_list]).set_index(['TICKER','DATE'])
    valid_data_list=[valid_data_list1,valid_data_list2]

    test_data_list = Parallel(n_jobs=15)(delayed(read_data)(date) for date in tqdm(test_list, desc='test datas are preparing'))
    test_data_list1 = pd.concat([x[0] for x in test_data_list]).set_index(['TICKER', 'DATE'])
    test_data_list2 = pd.concat([x[1] for x in test_data_list]).set_index(['TICKER', 'DATE'])
    test_data_list = [test_data_list1, test_data_list2]

    decision_tree_factors(train_data_list,valid_data_list,test_data_list)

if __name__=='__main__':

    run()