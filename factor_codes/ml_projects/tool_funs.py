import warnings
import feather
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# def group_mean(aa,file_name=''):
#     ret=feather.read_dataframe(r'C:\Users\admin\Desktop\ret.feather')
#     # ret.loc[ret['平均卖出收益_1100_1400'] > 0.4, '平均卖出收益_1100_1400'] = 0.4
#     ret['平均卖出收益_1100_1400'] = ret['平均卖出收益_1100_1400'].clip(-0.4, 0.4)  # 限制在[-0.4, 0.4]
#     ret['平均卖出收益_1100_1400'] -= 0.0007  # 统一减去一个基准值
#     ret['DATE'] = ret['DATE'].astype(str)
#     ret.rename(columns={'平均卖出收益_1100_1400':'ret'},inplace=True)
#     zz = pd.merge(ret, aa, how='inner', on=['TICKER', 'DATE'])
#     usename = list(np.setdiff1d(aa.columns, ['TICKER', 'DATE']))[0]
#     # bins = pd.qcut(zz[usename], 10)
#     zz['group'] = pd.qcut(zz[usename], 10, labels=False, duplicates='drop') + 1
#     res = zz.groupby('group')['ret'].mean().reset_index(drop=False)
#     print(res)
#     # res.to_csv(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\tyx_ml\{file_name}\test_res.csv',index=False)
#     return res

def split_df(df,y_col):
    warnings.filterwarnings('ignore')
    df.set_index(['TICKER','DATE'],inplace=True)
    y=df[[y_col]]
    x=df.drop(columns=y_col)
    return x,y

def ic_score(estimator,X,y):
    warnings.filterwarnings('ignore')
    y_pred=estimator.predict(X)
    return pearsonr(y.values.flatten(),y_pred.flatten())[0]

def normalize(df):
    # 数据预处理----------------------------------------------------------
    tar_columns=np.setdiff1d(df.columns,['TICKER','DATE','平均卖出收益_1100_1400'])
    tmp=df[['DATE','TICKER','平均卖出收益_1100_1400']]
    # df['tmp_use']=df['DATE']
    # df=df.set_index(['TICKER','DATE'])[tar_columns.tolist()+['tmp_use']].groupby('tmp_use').transform(lambda x:(x-x.mean())/x.std()).reset_index()
    df=df.set_index(['TICKER','DATE'])[tar_columns.tolist()].apply(lambda x:(x-x.mean())/x.std()).reset_index()
    df=df.merge(tmp,on=['DATE','TICKER'],how='inner')
    df.dropna(inplace=True)
    return df

def normalize1(df):
    # 数据预处理----------------------------------------------------------
    tar_columns=np.setdiff1d(df.columns,['TICKER','DATE','平均卖出收益_1100_1400'])
    tmp=df[['DATE','TICKER','平均卖出收益_1100_1400']]
    df['tmp_use']=df['DATE']
    # df=df.set_index(['TICKER','DATE'])[tar_columns.tolist()+['tmp_use']].groupby('tmp_use').transform(lambda x:(x-x.mean())/x.std()).reset_index()
    df=df.set_index(['TICKER','DATE'])[tar_columns.tolist()+['tmp_use']].groupby('tmp_use').transform(lambda x:(x-np.percentile(x,50))/(np.percentile(x,75)-np.percentile(x,25))).reset_index()
    # df=df.set_index(['TICKER','DATE'])[tar_columns.tolist()].apply(lambda x:(x-x.mean())/x.std()).reset_index()
    df=df.merge(tmp,on=['DATE','TICKER'],how='inner')
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.dropna(inplace=True)
    return df