# @Author: Yixin Tian
# @File: factor_min_data.py
# @Date: 2025/9/4 10:27
# @Software: PyCharm
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import read_min_data, update_muli, process_na_stock
import os
import feather
from joblib import Parallel,delayed

def get_data(date):
    warnings.filterwarnings('ignore')
    df=read_min_data(date)
    if df.empty:
        return
    df=df[['DATE','TICKER','min','volume','close']]
    df.sort_values(['TICKER','min'],inplace=True)
    df['ret_min'] = df.close.pct_change()
    df=df[(df['min']!=930)&(df['min']<1457)]
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.replace(np.nan,0,inplace=True)
    # 高频偏度：这里T选了一天的，日频
    tmp = df.groupby('TICKER').agg(sum_cube=('ret_min', lambda x: (x ** 3).sum()),sum_square=('ret_min', lambda x: (x ** 2).sum()),df_len=('ret_min',len))#.reset_index()
    # tmp1 = (np.sqrt(tmp['df_len']) * tmp['sum_cube']) / (tmp['sum_square'] ** 1.5)
    # tmp1=pd.DataFrame(tmp1).reset_index().rename(columns={0:'high_freq_skew'})
    tmp1=df.groupby('TICKER')['ret_min'].skew().reset_index().rename(columns={'ret_min':'high_freq_skew'})
    #上/下行波动占比
    tmp2=df.groupby('TICKER').agg(sum_square_low=('ret_min', lambda x: (x[x<0]**2).sum()),sum_square_up=('ret_min', lambda x: (x[x>0]**2).sum()))#.reset_index()
    tmp=tmp.merge(tmp2,right_index=True,left_index=True)
    tmp2=(np.sqrt(tmp['df_len']) * tmp['sum_square_low']) / tmp['sum_square']
    tmp2=pd.DataFrame(tmp2).reset_index().rename(columns={0:'low_vola_ratio'})
    tmp3=(np.sqrt(tmp['df_len']) * tmp['sum_square_up']) / tmp['sum_square']
    tmp3=pd.DataFrame(tmp3).reset_index().rename(columns={0:'up_vola_ratio'})
    # 量价相关性
    # df['vol_ratio']=df.groupby('TICKER')['volume'].transform(lambda x: x / x.sum())
    # tmp4=df.groupby('TICKER').agg(vol_price_corr=('close', lambda x: x.corr(df.loc[x.index, 'vol_ratio'])))
    # tmp4=pd.DataFrame(tmp4).reset_index()
    # 合并
    df=tmp1.merge(tmp2,on='TICKER',how='inner').merge(tmp3,on='TICKER',how='inner') #.merge(tmp4,on='TICKER',how='inner')
    df['DATE']=date
    df=df[['DATE','TICKER','high_freq_skew','low_vola_ratio','up_vola_ratio']] #,'vol_price_corr'
    # print(df)
    return df


def run(start='20200101',end='20221231'):
    calender = pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'))
    calender['trade_date'] = calender['trade_date'].astype(str)
    calender = calender[(calender.trade_date >= start) & (calender.trade_date <= end)]
    tar_date = sorted(list(calender.trade_date.unique()))
    tmp = Parallel(n_jobs=15)(delayed(get_data)(date) for date in tqdm(tar_date))
    tmp=pd.concat(tmp).reset_index(drop=True)
    tmp = process_na_stock(tmp,col='high_freq_skew')
    tmp.sort_values(['TICKER','DATE'],inplace=True)
    for col in tmp.columns[2:]:
        tmp[col+'_roll20']=tmp.groupby('TICKER')[col].rolling(20,5).mean().values
    tmp=tmp[['DATE','TICKER','high_freq_skew_roll20','low_vola_ratio_roll20','up_vola_ratio_roll20']]
    return [tmp]

def update(today='20250905'):
    update_muli('high_freq_skew_roll20.feather', today, run, num=-50)

if __name__=='__main__':
    # get_data('20250102')
    update()