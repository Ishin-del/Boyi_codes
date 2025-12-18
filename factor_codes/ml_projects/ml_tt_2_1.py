import os
import warnings
import feather
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib

# 数据集：20251110data_all.feather


def get_data():
    df=feather.read_dataframe(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\20251110data_all.feather')
    except_date=['20240206', '20240207', '20240208', '20240926', '20240927', '20240930', '20241008']
    df.dropna(inplace=True)
    special_df=df[df['DATE'].isin(except_date)]
    df=df[~df['DATE'].isin(except_date)]
    train_df=df[df['DATE']<='20240630']
    # print(train_df.describe())
    train_x,train_y=split_df(train_df,y_col='pf')
    valid_df=df[(df['DATE']>='20240701')&(df['DATE']<='20241231')]
    valid_x,vaild_y=split_df(valid_df,y_col='pf')
    test_df=df[df['DATE']>='20250101']
    test_x,test_y=split_df(test_df,y_col='pf')
    return train_x,train_y,valid_x,vaild_y,test_x,test_y

def split_df(df,y_col):
    warnings.filterwarnings('ignore')
    # df.rename(columns={'code':'TICKER','DATE':'DATE'},inplace=True)
    df.set_index(['TICKER','DATE'],inplace=True)
    y=df[[y_col]]
    x=df.drop(columns=y_col)
    return x,y


def ic_score(estimator,X,y):
    warnings.filterwarnings('ignore')
    y_pred=estimator.predict(X)
    return pearsonr(y.values.flatten(),y_pred.flatten())[0]

def ml():
    train_x,train_y,valid_x,valid_y,test_x,test_y=get_data()
    # model=XGBRegressor(random_state= 42,subsample= 0.8)
    grid_search=XGBRegressor(random_state= 42,n_jobs=12,subsample= 0.8,max_depth=3,min_child_weight=30,
                             gamma=0.15,learning_rate=0.1,reg_alpha=0.5,reg_lambda=0.5,n_estimators=300)
    # param_grid = {'max_depth':[3,4],
    #                'min_child_weight':[20,30,40],
    #               'gamma':[0.1,0.15],
    #                 'learning_rate':[0.05,0.1],
    #               'reg_alpha':[0.1,0.5],
    #               'reg_lambda':[0.1,0.5],
    #               'n_estimators':[500,1000]
    # }
    # grid_search = GridSearchCV(
    #     estimator=model,
    #     param_grid=param_grid,
    #     # cv=5,
    #     scoring=ic_score,
    #     n_jobs=12)
    grid_search.fit(train_x, train_y)
    # model.fit(train_x,train_y)
    train_pred_y=grid_search.predict(train_x)
    train=pd.DataFrame(train_pred_y, index=train_y.index, columns=['pred'])
    os.makedirs(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\tyx_ml\tyx_ml_2_1',exist_ok=True)
    feather.write_dataframe(train,r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\tyx_ml\tyx_ml_2_1\train.feather')
    group_mean(train)
    print('train ic:',np.corrcoef(train_pred_y.flatten(), train_y.values.flatten())[0,1])
    valid_pred_y = grid_search.predict(valid_x)
    valid = pd.DataFrame(valid_pred_y, index=valid_y.index, columns=['pred'])
    feather.write_dataframe(train,
                            r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\tyx_ml\tyx_ml_2_1\valid.feather')
    group_mean(valid)
    print('valid ic:', np.corrcoef(valid_pred_y.flatten(), valid_y.values.flatten())[0,1])
    test_pred_y = grid_search.predict(test_x)
    test = pd.DataFrame(test_pred_y, index=test_y.index, columns=['pred'])
    group_mean(test)
    feather.write_dataframe(train,
                            r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\tyx_ml\tyx_ml_2_1\test.feather')
    print('test ic:', np.corrcoef(test_pred_y.flatten(), test_y.values.flatten())[0,1])
    joblib.dump(grid_search, r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\tyx_ml\tyx_ml_2_1\xgb_model.joblib')
    # print('保存成功')

def group_mean(aa):
    ret=feather.read_dataframe(r'C:\Users\admin\Desktop\ret.feather')
    ret.loc[ret['平均卖出收益_1100_1400'] > 0.4, '平均卖出收益_1100_1400'] = 0.4
    ret['DATE'] = ret['DATE'].astype(str)
    ret.rename(columns={'平均卖出收益_1100_1400':'ret'},inplace=True)
    zz = pd.merge(ret, aa, how='inner', on=['TICKER', 'DATE'])
    usename = list(np.setdiff1d(aa.columns, ['TICKER', 'DATE']))[0]
    # bins = pd.qcut(zz[usename], 10)
    zz['group'] = pd.qcut(zz[usename], 10, labels=False, duplicates='drop') + 1
    res = zz.groupby('group')['ret'].mean().reset_index(drop=False)
    print(res)
    return res


if __name__=='__main__':
    ml()