# (也可以diff主动买或主动卖，看情绪变化激增或骤减的情况，这些区间挑选出来)
# order buy和实际成交之间的关系
import os

import feather
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import time
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import read_min_data, get_tar_date, process_na_stock


def cal(date):
    df=read_min_data(date)#[['TICKER','min','active_buy_volume','active_sell_volume','open','close']]
    # print(df.columns)
    df=df[(df['min']!=930)&(df['min']<1457)]
    df.sort_values(['TICKER', 'min'], inplace=True)
    df['min_ret']=df['close']/df['open']-1
    # df['min_ret']=(df['high'] - df['low']) / df['open']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # todo:

    df['amihud_illiquidity'] = abs(df['min_ret']) / df['amount']
    tmp = df.groupby(['TICKER'])['amihud_illiquidity'].mean().reset_index()
    tmp['DATE']=date
    tmp.columns=['TICKER','risk','DATE']
    return tmp

def compute_time_decay_weights(m, H):
    j_values = np.arange(1, m + 1)
    numerator = 2 ** (-(m - j_values + 1) / H)
    denominator = np.sum(2 ** (-j_values / H))
    weights = numerator / denominator
    return weights

def weighted_dot(x):
    weight = compute_time_decay_weights(len(x), int(len(x)/2))
    return np.dot(x, weight)

def run(start,end):
    risk_path=r'\\DESKTOP-79NUE61\Factor_Storage\Barra_日频'
    # date_list=get_tar_date(start,end)
    file_list=os.listdir(risk_path)
    res=pd.DataFrame()
    for file in file_list:
        tmp=feather.read_dataframe(os.path.join(risk_path,file))
        if res.empty:
            res=tmp.copy()
        else:
            res=res.merge(tmp,on=['DATE','TICKER'],how='inner')
    res=res[(res['DATE']>start)&(res['DATE']<end)]
    tar_columns=np.setdiff1d(res.columns,['DATE','TICKER'])
    for col in tar_columns:
        res[f'{col}_risk']=res.groupby('TICKER')[col].transform(lambda x: x.rolling(20, 5).mean())
        res[f'{col}_risk']=res[f'{col}_risk']-res[col]
        res.drop(columns=col,inplace=True)
    # print(res.columns)
    # --------------
    daily_df=feather.read_dataframe(DataPath.daily_path)
    daily_df.sort_values(['TICKER','DATE'],inplace=True)
    daily_df['daily_ret']=daily_df.groupby('TICKER')['close'].pct_change()
    daily_df=daily_df[(daily_df['DATE']>=start)&(daily_df['DATE']<=end)]
    daily_df=daily_df[['DATE','TICKER','daily_ret']]
    # --------------
    mkt_df=pd.read_csv(DataPath.wind_A_path)
    mkt_df['DATE']=mkt_df['DATE'].astype(str)
    mkt_df.sort_values(['DATE'],inplace=True)
    mkt_df['mkt_ret']=mkt_df['close'].pct_change()
    mkt_df=mkt_df[(mkt_df['DATE']>=start)&(mkt_df['DATE']<=end)][['DATE','mkt_ret']]
    # ---------------
    df=daily_df.merge(mkt_df,on='DATE',how='inner').merge(res,on=['DATE','TICKER'],how='inner')
    # df['risk_adjust']=1
    df=process_na_stock(df,'daily_ret')
    df.sort_values(['TICKER','DATE'],inplace=True)
    tar_columns2=np.setdiff1d(df.columns,['TICKER','DATE','mkt_ret','daily_ret'])
    for col in tar_columns2:
        df[f'{col}_factor']=df[col]*(df['daily_ret']-df['mkt_ret'])
        df[f'{col}_factor_adjust'] = df.groupby('TICKER')[f'{col}_factor'].rolling(20, 5).apply(weighted_dot,
                                                                raw=False).reset_index(level=0,drop=True)
        df.drop(columns=[col,f'{col}_factor'],inplace=True)
    # print(df.columns)
    df.drop(columns=['mkt_ret','daily_ret'],inplace=True)
    # df=df[['DATE','TICKER','factor_adjust']]
    # print(df)
    return [df]

def update_muli(filename,today,run,num=-50):
    if os.path.exists(os.path.join(DataPath.save_path_update,filename)):
    # if False:
        print('因子更新中')
        old=feather.read_dataframe(os.path.join(DataPath.save_path_update,filename))
        new_start=sorted(old.DATE.drop_duplicates().to_list())[num]
        res=run(start=new_start,end=today)
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                # feather.write_dataframe(tmp, os.path.join(r'C:\Users\admin\Desktop', col + '.feather'))
                old=feather.read_dataframe(os.path.join(DataPath.save_path_update,col+'.feather'))
                test=old.merge(tmp,on=['DATE','TICKER'],how='inner').dropna()
                test.sort_values(['TICKER','DATE'],inplace=True)
                tar_list = sorted(list(test.DATE.unique()))[-5:]
                test = test[test.DATE.isin(tar_list)]
                if np.isclose(test.iloc[:,2],test.iloc[:,3]).all():
                    tmp=tmp[tmp.DATE>old.DATE.max()]
                    old=pd.concat([old,tmp]).reset_index(drop=True).drop_duplicates()
                    print(old)
                    feather.write_dataframe(old,os.path.join(DataPath.save_path_update,col+'.feather'))
                    # feather.write_dataframe(old,os.path.join(DataPath.factor_out_path,col+'.feather'))
                else:
                    print(test[~np.isclose(test.iloc[:,2],test.iloc[:,3])])
                    # tt=test[~np.isclose(test.iloc[:, 2], test.iloc[:, 3])]
                    # feather.write_dataframe(tt,r'C:\Users\admin\Desktop\tt.feather')
                    print('检查更新出错!')
                    exit()
    else:
        print('因子生成中')
        # res=run(start='20200101',end='20221231')
        res=run(start='20211201',end='20250801')
        # res=run(start='20250828',end='20250829')
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                print(tmp)
                # feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                # feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))


def update(today):
    update_muli('factor.feather',today,run,-90)

if __name__=='__main__':
    # run('20250102','20250306')
    t1=time.time()
    update('20251014')
    t2=time.time()
    print((t2-t1)/60)