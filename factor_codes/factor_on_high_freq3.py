import os
import warnings
import feather
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import get_tar_date, read_min_data, update_muli
import time

# 20251028-中信建投-金融工程深度报告：高频流动性与波动率因子再构建

def min_cal(date):
    warnings.filterwarnings('ignore')
    df=read_min_data(date)
    df=df[(df['min']!=930)&(df['min']<1457)]
    for col in ['close', 'open', 'high', 'low']:
        df[f'log_{col}'] = np.log(df[col])
    df.sort_values(['TICKER', 'min'], inplace=True)
    df['pre_log_close']=df.groupby('TICKER')['log_close'].shift(1)
    df['pre_log_open']=df.groupby('TICKER')['log_open'].shift(1)
    df['log_close_diff']=df.groupby('TICKER')['log_close'].diff()
    df['pre_log_close_diff']=df.groupby('TICKER')['pre_log_close'].diff()
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.dropna(inplace=True)
    # Roll价差
    res1 = np.sqrt(np.maximum(-4*df.groupby('TICKER').apply(lambda x: np.cov(x['log_close_diff'], x['pre_log_close_diff'])[0, 1]), 0))
    # CS价差
    df['log_high_low']=np.log(df['high']/df['low'])
    df['pre_log_high_low']=df.groupby('TICKER')['log_high_low'].shift(1)
    df['pre_high']=df.groupby('TICKER')['high'].shift(1)
    df['pre_low']=df.groupby('TICKER')['low'].shift(1)
    df['max_high']=df[['high','low','pre_high','pre_low']].max(axis=1)
    df['min_low']=df[['high','low','pre_high','pre_low']].min(axis=1)
    df['gamma']=(np.log(df['max_high']/df['min_low']))**2
    df['beta']=((df['log_high_low'])**2+(df['pre_log_high_low'])**2)/2
    df['alpha']=(np.sqrt(2*df['beta'])-np.sqrt(df['beta']))/(3-2*np.sqrt(2))-np.sqrt(df['gamma']/(3-2*np.sqrt(2)))
    df['CS']=2*(np.exp(df['alpha'])-1)/(1+np.exp(df['alpha']))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    res2=df.groupby('TICKER')['CS'].mean()
    # AR
    df['midd_price']=(df['log_high']+df['log_low'])/2
    df.sort_values(['TICKER', 'min'], inplace=True)
    df['pre_close']=df.groupby('TICKER')['close'].shift(1)
    df['pre_open']=df.groupby('TICKER')['open'].shift(1)
    # df['close_diff']=df.groupby('TICKER')['close'].diff()
    df['pre_midd_price']=df.groupby('TICKER')['midd_price'].shift(1)
    df['pre_log_close']=df.groupby('TICKER')['log_close'].shift(1)
    df['AR']=np.sqrt(np.maximum(-4*(df['pre_log_close']-df['pre_midd_price'])*(df['pre_log_close']-df['midd_price']),0))
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    res3=df.groupby('TICKER')['AR'].mean()

    df['tau']=np.where((df['log_high'] == df['log_low']) & (df['log_low'] == df['pre_log_close']),0,1)
    P1 = df.groupby('TICKER').apply(lambda x: (x['log_open'] != x['log_high']).mean())
    P2 = df.groupby('TICKER').apply(lambda x: (x['log_open'] != x['log_low']).mean())
    P1_c = df.groupby('TICKER').apply(lambda x: (x['log_close'] != x['log_high']).mean())
    P2_c = df.groupby('TICKER').apply(lambda x: (x['log_close'] != x['log_low']).mean())
    tau_mean = df.groupby('TICKER')['tau'].mean()
    P1 *= tau_mean
    P2 *= tau_mean
    P1_c *= tau_mean
    P2_c *= tau_mean
    pi,pi_c = -8.0 / (P1 + P2),-8.0 / (P1_c + P2_c)
    # OHL
    tmp_data=df.groupby('TICKER').apply(OHL)
    S_square_OHL = pi * tmp_data
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    res4 = np.sqrt(S_square_OHL) * np.where(S_square_OHL > 0, 1, -1)
    # CHL
    tmp_data=df.groupby('TICKER').apply(CHL)
    S_square_CHL=pi_c*tmp_data
    res10=np.sqrt(S_square_CHL) * np.where(S_square_CHL > 0, 1, -1)
    # OHLC
    tmp_data=df.groupby('TICKER').apply(OHLC)
    S_square_OHLC=pi*tmp_data
    res11=np.sqrt(S_square_OHLC) * np.where(S_square_OHLC > 0, 1, -1)
    # CHLO
    tmp_data=df.groupby('TICKER').apply(CHLO)
    S_square_CHLO=pi_c*tmp_data
    res12=np.sqrt(S_square_CHLO) * np.where(S_square_OHLC > 0, 1, -1)
    # EDGE
    S_square_EDGE=df.groupby('TICKER').apply(EDGE,pi,pi_c)
    res13=np.sqrt(S_square_EDGE)*np.where(S_square_EDGE > 0, 1, -1)
    # 波动率因子========================================================
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    res5=np.sqrt(df.groupby('TICKER').apply(lambda x: ((np.log(x['high']/x['low']))**2).mean()*1/4*np.log(2))) #Parkinson
    res6=np.sqrt(df.groupby('TICKER').apply(Garman_Klass))#Garman_Klass
    res7=np.sqrt(df.groupby('TICKER').apply(Rogers_Satchell)) #
    df=df.merge(res7.reset_index(),on='TICKER',how='left').rename(columns={0:'sigma_RS'})
    res8=np.sqrt(df.groupby('TICKER').apply(yang_zhang)) #Yang-Zhang
    res9=np.sqrt(df.groupby('TICKER').apply(cho_free))
    res=[res1,res2,res3,res4,res5,res6,res7,res8,res9,res10,res11,res12,res13]
    res=pd.concat(res,axis=1).reset_index()
    res['DATE']=date
    res.columns=['TICKER','ROLL','CS','AR','OHL','Parkinson','Garman_Klass','Rogers_Satchell','Yang_Zhang','Cho_Frees','CHL','OHLC','CHLO','EDGE','DATE']
    res=res[['TICKER','DATE','ROLL','CS','AR','OHL','Parkinson','Garman_Klass','Rogers_Satchell','Yang_Zhang','Cho_Frees','CHL','OHLC','CHLO','EDGE']]
    return res

def Garman_Klass(x):
    num1=np.log(x['high'] / x['low']) ** 2
    num2=np.log(x['close'] / x['open']) ** 2
    num3=2 * np.log(2) - 1
    return (0.5 *num1 - num3 * num2).mean()

def Rogers_Satchell(x):
    num1=np.log(x['high'] / x['close'])
    num2=np.log(x['high'] / x['open'])
    num3=np.log(x['low'] / x['close'])
    num4=np.log(x['low'] / x['open'])
    return (num1*num2+num3*num4).mean()

def cho_free(x):
    num=np.log(1+1/8)
    miu=x['pre_log_close'].mean()/2*num
    tao=x['pre_log_close'].mean()/len(x)
    return 2*miu*num/(np.log(num+miu*tao)-np.log(num-miu*tao))

def yang_zhang(x):
    # todo: alph=0.34
    sigma_o=np.abs(x['log_open']-x['pre_log_close']-1).sum()
    try:
        k=(0.34/(1.34+(len(x)+1)/(len(x)-1)))
    except ZeroDivisionError:
        return np.nan
    sigma_c=np.abs(x['log_close']-x['pre_log_open']-1).sum()
    sigma_RS=(x['sigma_RS'].iloc[0])**2
    return sigma_o+k*sigma_c+(1-k)*sigma_RS

def EDGE(group,pi,pi_c):
    # todo:这个应该有点不对
    pi=pi[pi.index==group['TICKER'].iloc[0]].iloc[0]
    pi_c=pi_c[pi_c.index==group['TICKER'].iloc[0]].iloc[0]
    x1=pi/2*(group['midd_price'] - group['log_open']) * (group['log_open'] - group['pre_midd_price'])+pi_c/2*(group['midd_price'] - group['pre_log_close']) * (group['pre_log_close'] - group['pre_midd_price'])
    x2=pi/2*(group['midd_price'] - group['log_open']) * (group['log_open'] - group['pre_log_close'])+pi_c/2*((group['log_open'] - group['pre_log_close']) * (group['pre_log_close'] - group['pre_midd_price']))
    try:
        w1=np.var(x2)/(np.var(x1)+np.var(x2))
        w2=np.var(x1)/(np.var(x1)+np.var(x2))
    except ZeroDivisionError:
        return np.nan
    return w1*np.mean(x1)+w2*np.mean(x2)

def OHL(group): # OHL
    tmp1 = ((group['midd_price'] - group['log_open']) * (group['log_open'] - group['pre_midd_price'])).mean()
    tmp2 = (group['midd_price'] - group['log_open']).mean()
    tmp3 = ((group['log_open'] - group['pre_midd_price']) * group['tau']).mean()
    tau_mean = group['tau'].mean()
    return tmp1 - (tmp2 * tmp3) / tau_mean

def CHL(group): # CHL
    tmp1 = ((group['midd_price'] - group['pre_log_close']) * (group['pre_log_close'] - group['pre_midd_price'])).mean()
    tmp2 = (group['midd_price'] - group['pre_log_close']).mean()
    tmp3 = ((group['pre_log_close'] - group['pre_midd_price']) * group['tau']).mean()
    tau_mean = group['tau'].mean()
    return tmp1 - (tmp2 * tmp3) / tau_mean

def OHLC(group): #OHLC
    tmp1 = ((group['midd_price'] - group['log_open']) * (group['log_open'] - group['pre_log_close'])).mean()
    tmp2 = (group['midd_price'] - group['log_open']).mean()
    tmp3 = ((group['log_open'] - group['pre_log_close']) * group['tau']).mean()
    tau_mean = group['tau'].mean()
    return tmp1 - (tmp2 * tmp3) / tau_mean

def CHLO(group): #CHLO
    tmp1 = ((group['log_open'] - group['pre_log_close']) * (group['pre_log_close'] - group['pre_midd_price'])).mean()
    tmp2 = (group['log_open'] - group['pre_log_close']).mean()
    tmp3 = ((group['pre_log_close'] - group['pre_midd_price']) * group['tau']).mean()
    tau_mean = group['tau'].mean()
    return tmp1 - (tmp2 * tmp3) / tau_mean

def run(start,end):
    tar_date=get_tar_date(start,end)
    # df=[]
    # for date in tqdm(tar_date):
    #     tmp=min_cal(date)
    #     df.append(tmp)
    df=Parallel(n_jobs=13)(delayed(min_cal)(date) for date in tqdm(tar_date))
    df=pd.concat(df).reset_index(drop=True)
    # daily_cal
    daily_df = feather.read_dataframe(DataPath.daily_path)
    daily_df['amount']/=10000
    df = df.merge(daily_df[['DATE','TICKER','amount']], on=['DATE', 'TICKER'], how='inner')
    # 除以amount
    tar_col=np.setdiff1d(df.columns,['DATE','TICKER','amount'])
    for col in tar_col:
        df[f'{col}_I']=df[col]/df['amount']
    df.drop(columns='amount',inplace=True)
    # df.to_csv(r'C:\Users\admin\Desktop\check.csv')
    print(df)
    return [df]

def update(today='20251103'):
    update_muli('Yang_Zhang_I.feather',today,run)

if __name__=='__main__':
    t1=time.time()
    update()
    t2=time.time()
    print((t2-t1)/60)