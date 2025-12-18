# @Author: Yixin Tian
# @File: ml_factor.py
# @Date: 2025/9/8 10:21
# @Software: PyCharm

import feather
import numpy as np
import pandas as pd
from tool_tyx.tyx_funcs import get_ml_data
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler

def group_mean(aa,path=r"\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\Exit_4_data_4个时段_orderbook_beta_722.xlsx"):
    ret = pd.read_excel(path)
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

def preprocess_data():

    pass

def model_test(train_data_list,valid_data_list,test_data_list): #
    # scaler = StandardScaler()
    # 训练
    train_x, train_y =train_data_list[0].apply(lambda x: (x-x.mean())/x.std(),axis=1), train_data_list[1]  # train_x,train_x
    valid_x, valid_y = valid_data_list[0].apply(lambda x: (x-x.mean())/x.std(),axis=1), valid_data_list[1]
    test_x, test_y = test_data_list[0].apply(lambda x: (x-x.mean())/x.std(),axis=1), test_data_list[1]

    # todo:模型：
    model = Lasso(alpha=0.01, fit_intercept=True, tol=1e-4, max_iter=10000)
    # model = LassoCV(cv=10)
    model.fit(train_x, train_y)
    # 训练数据-----------------------------------------------------
    pre_train_y=model.predict(train_x)
    pre_train_y = pd.DataFrame(pre_train_y, index=train_y.index, columns=['pred_train'])  # .reset_index()
    # feather.write_dataframe(pre_train_y,r'C:\Users\admin\Desktop\pred_train.feather')
    # print('train_res:',pre_train_y)
    group_mean(pre_train_y)
    # print('------------------------')
    # 验证数据-----------------------------------------------------
    pre_valid_y = model.predict(valid_x)
    # print(f'ic on testdata: {np.corrcoef(pre_valid_y, valid_y)[0][1]}')
    pre_valid_y = pd.DataFrame(pre_valid_y, index=valid_y.index, columns=['pred_valid'])  # .reset_index()
    # feather.write_dataframe(pre_valid_y, r'C:\Users\admin\Desktop\pred_valid.feather')
    # print('valid_res:',pre_valid_y)
    group_mean(pre_valid_y)
    # print(model.alpha_ )
    # print('------------------------')
    # 测试数据-----------------------------------------------------
    pre_test_y = model.predict(test_x)
    pre_test_y = pd.DataFrame(pre_test_y, index=test_y.index, columns=['pred_test'])#.reset_index()
    # feather.write_dataframe(pre_test_y, r'C:\Users\admin\Desktop\pred_test.feather')
    # print(pre_test_y)
    # print('test_res:',pre_test_y)
    group_mean(pre_test_y)


def run(path):
    train_data_list,valid_data_list,test_data_list=get_ml_data(path)
    model_test(train_data_list, valid_data_list, test_data_list)
    # for i in np.linspace(start = 0, stop = 1, num = 10)[1:-1]:
    #     print(round(i,1))
    #     model_test(train_data_list,valid_data_list,test_data_list,alpha=round(i,1))
    #     print('---------------')

if __name__=='__main__':
    run(path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\板二底表20251014\train')

