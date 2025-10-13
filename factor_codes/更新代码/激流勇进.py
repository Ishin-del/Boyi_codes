# @Author: Yixin Tian
# @File: 激流勇进.py
# @Date: 2025/9/9 10:07
# @Software: PyCharm
import os
import warnings

import feather
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import read_min_data, process_na_stock, get_tar_date
import numpy as np

def cal(date):
    warnings.filterwarnings('ignore')
    df=read_min_data(date)
    if df.empty:
        return
    df=df[(df['min']!=930)&(df['min']<1457)][['DATE', 'TICKER', 'min', 'open', 'high', 'low', 'close', 'volume','amount']]
    df=process_na_stock(df,col='close')
    df.sort_values(['TICKER','min'],inplace=True)
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df['vol_5min']=df.groupby('TICKER').volume.rolling(5).sum().values #过去5分钟收益率,先计算个股每分钟的成交量及其之前4分钟成交量的总和
    df['vol_diff'] = df.groupby('TICKER').volume.diff() #每分钟较前一分钟进行判断
    df['vol_stat'] = np.where(df['vol_diff'] > 0, 'expand', 'reduce')  # 根据成交量变化 判断当前分钟为“放量”还是“缩量”
    # 对每分钟，依据过去5分钟内高、开、低、收数据，计算近期收益率趋势，趋势为正则为“上涨”状态，反之为“下跌”状态
    df['price_open_5min'] = df.groupby('TICKER')['open'].shift(4)
    df['price_total']=df[['open', 'high', 'low', 'close']].sum(axis=1)
    df['price_mean_5min']=(df.groupby('TICKER')['price_total'].rolling(5).sum()/20).values
    df['ret_stat']=np.where(df['price_mean_5min']>df['price_open_5min'],'rise', 'fell')
    df['label']=df['vol_stat']+'_'+df['ret_stat'] #四类：放量上涨、放量下跌、缩量上涨、缩量下跌
    # 分组均值/日内日均值
    tmp=df.groupby('TICKER')['amount'].mean().reset_index().rename(columns={'amount':'amount_mean'})
    tmp1=df.groupby(['TICKER','label'])['amount'].mean().reset_index().rename(columns={'amount':'label_mean'})
    tmp=tmp.merge(tmp1,on='TICKER',how='right')
    tmp['amount_ratio']=tmp['label_mean']/tmp['amount_mean']
    tmp1=df.groupby('TICKER')['volume'].mean().reset_index().rename(columns={'volume':'volume_mean'})
    tmp2=df.groupby(['TICKER','label'])['volume'].mean().reset_index().rename(columns={'volume':'label_mean'})
    tmp1= tmp1.merge(tmp2, on='TICKER', how='right')
    tmp1['volume_ratio']=tmp1['label_mean']/tmp1['volume_mean']
    tmp=tmp[['TICKER','label','amount_ratio']].merge(tmp1[['TICKER','label','volume_ratio']],on=['TICKER','label'],how='inner')
    # 金额-量，做差：逻辑在于，如果成交金额比例明显高于成交量比例，说明investor的买入意愿更强
    tmp['buy_str'] = tmp['amount_ratio'] - tmp['volume_ratio']
    tmp=tmp.pivot(index='TICKER',columns='label',values='buy_str').reset_index()
    tmp['DATE']=date
    return tmp[['DATE','TICKER','expand_fell', 'expand_rise', 'reduce_fell', 'reduce_rise']]

def run(start,end):
    warnings.filterwarnings('ignore')
    tar_list=get_tar_date(start,end)
    df=Parallel(n_jobs=15)(delayed(cal)(date) for date in tqdm(tar_list))
    df=pd.concat(df)
    df=process_na_stock(df,col='expand_fell')
    res=[]
    for col in df.columns.tolist()[2:]:
        tmp=df[['DATE','TICKER',col]]
        tmp[col+'_roll20']=tmp.groupby('TICKER')[col].rolling(20,5).mean().values
        tmp=tmp[['DATE','TICKER',col+'_roll20']]
        res.append(tmp)
    return res

def update(today='20250905'):
    update_muli('reduce_rise_roll20.feather', today, run,num=-70)

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
                    feather.write_dataframe(old,os.path.join(DataPath.factor_out_path,col+'.feather'))
                else:
                    print(test[~np.isclose(test.iloc[:,2],test.iloc[:,3])])
                    # tt=test[~np.isclose(test.iloc[:, 2], test.iloc[:, 3])]
                    # feather.write_dataframe(tt,r'C:\Users\admin\Desktop\tt.feather')
                    print('检查更新出错!')
                    exit()
    else:
        print('因子生成中')
        res=run(start='20200101',end='20221231')
        # res=run(start='20200101',end='20250822')
        # res=run(start='20250828',end='20250829')
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                print(tmp)
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))
                # feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))

if __name__=='__main__':
    update()