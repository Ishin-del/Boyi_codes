import feather
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
from tool_tyx.path_data import DataPath


def process_na_value(df):
    '''解决某些股票出现时间序列断开的情况，这种情况会在rolling（window!=min_period）的时候,因子值出现不同'''
    pt = df.pivot_table(columns='TICKER', index='DATE', values='close')
    pt = pt.ffill()
    pp = pt.melt(ignore_index=False).reset_index(drop=False)
    df = pd.merge(pp, df, how='left', on=['TICKER', 'DATE'])
    use_cols = list(np.setdiff1d(df.columns, ['TICKER', 'DATE']))
    for c in use_cols:
        df[c] = df.groupby('TICKER')[c].ffill()
    return df

def cal_ctr_fun(i, full_dl, df, window=20):
    warnings.filterwarnings('ignore')
    tmp_date_list = full_dl[i:i + window]
    tmp = df[df.DATE.isin(tmp_date_list)][['DATE', 'TICKER', 'turnover', 'over_night_smart']]
    tmp.sort_values(['TICKER', 'DATE'], inplace=True)
    tmp = tmp.groupby('TICKER').filter(lambda x: len(x) >= 20)  # 每个循环 去掉不够20天的
    date_df = tmp.groupby('TICKER', as_index=False)['DATE'].last()
    tmp['turnover_shift1'] = tmp.groupby('TICKER')['turnover'].shift(1)
    tmp.sort_values(['TICKER', 'over_night_smart'], inplace=True)
    tmp = tmp.groupby('TICKER').head(4)
    tmp_res = tmp.groupby('TICKER', as_index=False)['turnover_shift1'].mean().rename(columns={'turnover_shift1': 'ctr'})
    # 选中当天的数据直接对前天的换手求均值
    tmp = tmp_res.merge(date_df, on='TICKER', how='left')
    tmp = tmp[['DATE', 'TICKER', 'ctr']]
    return tmp

def helper(date, sh_path=DataPath.sh_min, sz_path=DataPath.sz_min):
    try:
        sh_data = feather.read_dataframe(os.path.join(sh_path, f'{date}.feather'))
        sz_data = feather.read_dataframe(os.path.join(sz_path, f'{date}.feather'))
    except FileNotFoundError:
        return
    tmp_sh = sh_data[sh_data['min'] == 930][['TICKER', 'volume']]
    tmp_sh['volume'] *= 100
    tmp_sh.columns = ['TICKER', 'Qty']
    tmp_sh['DATE'] = date
    tmp_sz = sz_data[sz_data['min'] == 930][['TICKER', 'volume']]
    tmp_sz['volume'] *= 100
    tmp_sz.columns = ['TICKER', 'Qty']
    tmp_sz['DATE'] = date
    return pd.concat([tmp_sh, tmp_sz])
# todo:
def get_early_vol(full_dl, file_path=os.path.join(DataPath.tmp_path,'集合竞价成交量.feather')):
    start, end = min(full_dl), max(full_dl)
    if os.path.exists(file_path):  # 存在则直接读取
    # f=1
    # if f==2:
        tmp = feather.read_dataframe(file_path)
        if max(tmp.DATE) < end:  # 时间不够，则生成数据,min(tmp.DATE) > start or
            cal_dl = list(set(full_dl)-set(tmp.DATE))
            # date_code=date_code[~date_code.DATE.isin(tmp.DATE)]
            print('早盘集合竞价成交量数据计算&保存:')
            tmp1 = Parallel(n_jobs=12)(delayed(helper)(data) for data in tqdm(cal_dl))
            tmp1 = pd.concat(tmp1).reset_index(drop=True)
            tmp = pd.concat([tmp, tmp1]).reset_index(drop=True)
            feather.write_dataframe(tmp, file_path)
    else:  # 不存在则生成
        print('早盘集合竞价成交量数据计算&保存:')
        tmp = Parallel(n_jobs=12)(delayed(helper)(data) for data in tqdm(full_dl))
        tmp = pd.concat(tmp).reset_index(drop=True)
        feather.write_dataframe(tmp, file_path)
    return tmp

def read_data(time_start='',time_end='',
              df_path=DataPath.daily_path,
              total_share_path=os.path.join(DataPath.to_data_path,'totalShares.h5'),
              adj_df_path=os.path.join(DataPath.to_df_path,'adj_factors.feather')
              ):
    df = feather.read_dataframe(df_path)[['DATE', 'TICKER', 'open', 'high', 'low', 'close', 'volume']]
    # df = df[df.TICKER == '000430.SZ']
    df = df[(df['DATE'] <= time_end) & (df['DATE'] >= time_start)]
    # date_code=df[['TICKER','DATE']].drop_duplicates()
    full_dl = df.DATE.drop_duplicates().to_list()
    full_dl =sorted(full_dl)
    # print('1')
    tmp = pd.read_hdf(total_share_path).reset_index()
    tmp.time=tmp.time.str.replace('-','')
    tmp=tmp[(tmp.time<=time_end) & (tmp.time>=time_start)]
    tmp.rename(columns={'time': 'DATE'}, inplace=True)
    tmp.DATE = tmp.DATE.str.replace('-', '')
    tmp = tmp.melt(id_vars=['DATE'], var_name='TICKER', value_name='total_share')
    df = df.merge(tmp, how='left', on=['DATE', 'TICKER'])

    tmp = feather.read_dataframe(adj_df_path)
    df = df.merge(tmp, how='left', on=['DATE', 'TICKER'])
    df = process_na_value(df)
    # 复权价格
    df['close'] *= df['adj_factors']
    df['open'] *= df['adj_factors']
    # 市值因子
    df['mkt_size'] = df['total_share'] * df['close']
    df.sort_values(['TICKER', 'DATE'], inplace=True)

    # 拉取早盘成交量
    tmp=get_early_vol(full_dl)
    df=df.merge(tmp,how='left', on=['DATE', 'TICKER'])
    df['pre_close'] = df.groupby('TICKER')['close'].shift(1).values
    df['intra_ret'] = df['close'] / df['open'] - 1  # 日内收益率
    df['night_ret'] = df.groupby('TICKER')['close'].shift(1) / df['open'] - 1  # 隔夜收益率
    # 换手率计算
    df['turnover'] = df['volume'] * 100 / df['total_share']  # 日内换手率
    df['pre_turnover'] = df.groupby('TICKER')['turnover'].shift(1).values  # 昨日换手率
    df['pre_total_share'] = df.groupby('TICKER')['total_share'].shift(1)  # 昨日总股本
    df['overnight_turn'] = df['Qty'] * 100 / df['pre_total_share']  # 隔夜换手率
    df.drop(columns=['total_share','adj_factors','Qty'], inplace=True)
    return df, full_dl


def cal(df, full_dl,window=20):
    warnings.filterwarnings(action='ignore')
    df.sort_values(['TICKER', 'DATE'], inplace=True)
    print('OvernightSmart20因子计算中：')
    df['night_turn_min'] = df.groupby('TICKER')['night_ret'].rolling(window=20, min_periods=10).min().values
    df['night_turn_max'] = df.groupby('TICKER')['night_ret'].rolling(window=20, min_periods=10).max().values
    df['smart_tmp'] = (df['night_ret'] - df['night_turn_min']) / (df['night_turn_max'] - df['night_turn_min'])
    df['over_night_smart'] = df['smart_tmp'] / df['overnight_turn']
    df['OvernightSmart20'] = df.groupby('TICKER')['over_night_smart'].rolling(window=20, min_periods=10).mean().values

    # ctr因子计算--------------------------------------------------------------------------------------------------
    df['pre_over_night_smart'] = df.groupby('TICKER')['over_night_smart'].shift(-1)  # 次日隔夜聪明钱
    print('ctr因子计算中：')
    ctr = Parallel(n_jobs=13)(delayed(cal_ctr_fun)(i,full_dl,df) for i in tqdm(range(len(full_dl) - window + 1)))
    ctr = pd.concat(ctr).reset_index(drop=True)

    return df[['TICKER', 'DATE', 'OvernightSmart20']],ctr


if __name__ == '__main__':
    pass