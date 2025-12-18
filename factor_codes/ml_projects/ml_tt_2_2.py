import os
import warnings
import feather
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import lightgbm as lgb
import joblib

# 数据集：20251110data_all.feather


def get_data():
    df=feather.read_dataframe(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\20251110data_all.feather')
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df=df[['TICKER','DATE','158', '159', '135', '124', '111', '89', '57', '10','pf']]
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
    # df.rename(columns={'code':'TICKER','DATE':'DATE'},inplace=True)               verbose=-1
    df.set_index(['TICKER','DATE'],inplace=True)
    y=df[[y_col]]
    x=df.drop(columns=y_col)
    return x,y


def ic_score(estimator,X,y):
    warnings.filterwarnings('ignore')
    y_pred=estimator.predict(X)
    return pearsonr(y.values.flatten(),y_pred.flatten())[0]


def ml(file_name):
    path = fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\tyx_ml\{file_name}'
    train_x,train_y,valid_x,valid_y,test_x,test_y=get_data()
    model = lgb.LGBMRegressor(random_state=42,verbose=-1)
    param_grid = {
        'n_estimators': [500],
        'learning_rate': [0.01,0.02, 0.05],
        'max_depth': [3],
        'num_leaves': [5,7],
        'subsample': [0.8],
        'reg_lambda':[0.1,0.8],
        'reg_alpha':[0.1,0.8]
    }
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        scoring=ic_score,
        n_jobs=12)
    grid_search.fit(train_x, train_y)
    print("最佳参数:", grid_search.best_params_)
    print("最佳IC分数:", grid_search.best_score_)
    train_pred_y=grid_search.predict(train_x)
    train=pd.DataFrame(train_pred_y, index=train_y.index, columns=['pred'])
    group_mean(train)
    os.makedirs(path,exist_ok=True)
    feather.write_dataframe(train,os.path.join(path,'train.feather'))
    print('train ic:',pearsonr(train_pred_y.flatten(), train_y.values.flatten())[0])

    valid_pred_y = grid_search.predict(valid_x)
    valid = pd.DataFrame(valid_pred_y, index=valid_y.index, columns=['pred'])
    group_mean(valid)
    feather.write_dataframe(train,os.path.join(path,'valid.feather'))

    print('valid ic:',pearsonr(valid_pred_y.flatten(), valid_y.values.flatten())[0])
    test_pred_y = grid_search.predict(test_x)
    test = pd.DataFrame(test_pred_y, index=test_y.index, columns=['pred'])
    group_mean(test)
    feather.write_dataframe(train,os.path.join(path,'test.feather'))
    print('test ic:', pearsonr(test_pred_y.flatten(), test_y.values.flatten())[0])
    joblib.dump(model, os.path.join(path,'lgb_model.joblib'))
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