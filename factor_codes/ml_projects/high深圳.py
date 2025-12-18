import os
import warnings
import feather
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
from tool_funs import *
# 数据集：C:\Users\admin\Desktop\high深圳.feather


def get_data(file_name):
    df=feather.read_dataframe(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\吴文博alpha_pool\{file_name}.feather')
    df.drop(columns=['exit_4_return','auction_amount','开盘后10秒vwap收益_1100_1400','开盘后30秒vwap收益_1100_1400',
                     '开盘后60秒vwap收益_1100_1400','开盘后180秒vwap收益_1100_1400','开盘后300秒vwap收益_1100_1400',
                     '开盘后600秒vwap收益_1100_1400','开盘后1200秒vwap收益_1100_1400','开盘后1800秒vwap收益_1100_1400'],
            inplace=True)

    except_date=['20240206', '20240207', '20240208', '20240926', '20240927', '20240930', '20241008']
    df.dropna(inplace=True)
    df['平均卖出收益_1100_1400'] = df['平均卖出收益_1100_1400'].clip(-0.4, 0.4)  # 限制在[-0.4, 0.4]
    df['平均卖出收益_1100_1400'] -= 0.0007  # 统一减去一个基准值
    special_df=df[df['entry_date'].isin(except_date)]
    df=df[~df['entry_date'].isin(except_date)]
    train_df=df[df['entry_date']<='20240630']
    train_x,train_y=split_df(train_df)
    valid_df=df[(df['entry_date']>='20240701')&(df['entry_date']<='20241231')]
    valid_x,vaild_y=split_df(valid_df)
    test_df=df[df['entry_date']>='20250101']
    test_x,test_y=split_df(test_df)
    return train_x,train_y,valid_x,vaild_y,test_x,test_y

def split_df(df):
    warnings.filterwarnings('ignore')
    df.rename(columns={'code':'TICKER','entry_date':'DATE'},inplace=True)
    df.set_index(['TICKER','DATE'],inplace=True)
    y=df[['平均卖出收益_1100_1400']]
    x=df.drop(columns='平均卖出收益_1100_1400')
    return x,y

def ml(file_name):
    train_x,train_y,valid_x,valid_y,test_x,test_y=get_data(file_name)
    """
    备胎1
    model = XGBRegressor(n_estimators=1000,max_depth=6,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,
                         random_state=42,n_jobs=12)
    备胎2（不如胎1好）
    model=LGBMRegressor(n_estimators=1000,max_depth=6,learning_rate=0.05,random_state=42,n_jobs=12)                   
    """
    model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         random_state=42, n_jobs=12)
    model.fit(train_x,train_y)
    train_pred_y=model.predict(train_x)
    train=pd.DataFrame(train_pred_y, index=train_y.index, columns=['pred'])
    # group_mean(train)
    print('train ic:',np.corrcoef(train_pred_y.flatten(), train_y.values.flatten())[0,1])
    valid_pred_y = model.predict(valid_x)
    valid = pd.DataFrame(valid_pred_y, index=valid_y.index, columns=['pred'])
    # group_mean(valid)
    print('valid ic:', np.corrcoef(valid_pred_y.flatten(), valid_y.values.flatten())[0,1])
    test_pred_y = model.predict(test_x)
    test = pd.DataFrame(test_pred_y, index=test_y.index, columns=['pred'])
    path=fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\tyx_ml\{file_name}'

    os.makedirs(path, exist_ok=True)
    group_mean(test,file_name)
    print('test ic:', np.corrcoef(test_pred_y.flatten(), test_y.values.flatten())[0,1])

    os.makedirs(path,exist_ok=True)
    feather.write_dataframe(train, os.path.join(path, 'train.feather'))
    feather.write_dataframe(valid, os.path.join(path, 'valid.feather'))
    feather.write_dataframe(test, os.path.join(path, 'test.feather'))
    joblib.dump(model,  os.path.join(path,'xgb.joblib'))
    print('保存成功')



if __name__=='__main__':
    ml('high深圳')