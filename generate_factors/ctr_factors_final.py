import feather
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings


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

def cal_jump_ctr_fun(i, full_dl, df, window=20):
    tmp_date_list = full_dl[i:i + window]
    tmp = df[df.DATE.isin(tmp_date_list)][['DATE', 'TICKER', 'turnover', 'pre_over_night_smart']]
    tmp.sort_values(['TICKER', 'DATE'], inplace=True)
    tmp = tmp.groupby('TICKER').filter(lambda x: len(x) >= 20)
    last_day_turn = tmp.groupby('TICKER')['turnover'].last().reset_index().rename(columns={'turnover': 'tmp_turn'})
    date_df = tmp.groupby('TICKER', as_index=False)['DATE'].last()
    tmp.sort_values(['TICKER', 'pre_over_night_smart'], inplace=True)
    tmp_res = tmp.groupby('TICKER')[['TICKER', 'turnover']].head(3).rename(columns={'turnover': 'tmp_turn'})
    tmp_res = pd.concat([tmp_res, last_day_turn])
    tmp_res = tmp_res.groupby('TICKER')['tmp_turn'].mean().reset_index()
    tmp_res.rename(columns={'tmp_turn': 'jump_ctr'}, inplace=True)
    tmp = tmp_res.merge(date_df, on='TICKER', how='left')
    tmp = tmp[['DATE', 'TICKER', 'jump_ctr']]
    return tmp

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

def helper(date, sh_path=r'\\DESKTOP-79NUE61\SH_min_data', sz_path=r'\\DESKTOP-79NUE61\SZ_min_data'):
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
    # return tmp_sz

def get_early_vol(full_dl, file_path=r'D:\tyx\raw_data\集合竞价成交量.feather'):
    start, end = min(full_dl), max(full_dl)
    if os.path.exists(file_path):  # 存在则直接读取
        tmp = feather.read_dataframe(file_path)
        if min(tmp.DATE) > start or max(tmp.DATE) < end:  # 时间不够，则生成数据
            cal_dl = list(set(full_dl)-set(tmp.DATE))
            print('早盘集合竞价成交量数据计算&保存:')
            tmp1 = Parallel(n_jobs=12)(delayed(helper)(date) for date in tqdm(cal_dl))
            tmp1 = pd.concat(tmp1).reset_index(drop=True)
            tmp = pd.concat([tmp, tmp1]).reset_index(drop=True)
            feather.write_dataframe(tmp, file_path)
    else:  # 不存在则生成
        print('早盘集合竞价成交量数据计算&保存:')
        tmp = Parallel(n_jobs=12)(delayed(helper)(date) for date in tqdm(full_dl))
        tmp = pd.concat(tmp).reset_index(drop=True)
        feather.write_dataframe(tmp, file_path)
    return tmp

def read_data(time_start='',time_end='',
              df_path=r'D:\tyx\raw_data\daily.feather',
              total_share_path=r'D:\tyx\raw_data\totalShares.h5',
              adj_df_path=r'Z:\local_data\Data_Storage\adj_factors.feather'
              ):
    df = feather.read_dataframe(df_path)[['DATE', 'TICKER', 'open', 'high', 'low', 'close', 'volume']]
    # 仅作测试用
    # df = df[df.TICKER == '000004.SZ']
    df = df[(df['DATE'] <= time_end) & (df['DATE'] >= time_start)]
    full_dl = df.DATE.drop_duplicates().to_list()
    sorted(full_dl)

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


def cal(df, full_dl, save_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\2020-2022.12',
        window=20):
    warnings.filterwarnings(action='ignore')
    df.sort_values(['TICKER', 'DATE'], inplace=True)
    print('OvernightSmart20因子计算中：')
    df['night_turn_min'] = df.groupby('TICKER')['night_ret'].rolling(window=20, min_periods=10).min().values
    df['night_turn_max'] = df.groupby('TICKER')['night_ret'].rolling(window=20, min_periods=10).max().values
    df['smart_tmp'] = (df['night_ret'] - df['night_turn_min']) / (df['night_turn_max'] - df['night_turn_min'])
    df['over_night_smart'] = df['smart_tmp'] / df['overnight_turn']
    df['OvernightSmart20'] = df.groupby('TICKER')['over_night_smart'].rolling(window=20, min_periods=10).mean().values
    # feather.write_dataframe(df[['TICKER', 'DATE', 'OvernightSmart20']], os.path.join(save_path, 'OvernightSmart20.feather'))

    # ctr因子计算--------------------------------------------------------------------------------------------------
    df['pre_over_night_smart'] = df.groupby('TICKER')['over_night_smart'].shift(-1)  # 次日隔夜聪明钱
    print('ctr因子计算中：')
    ctr = Parallel(n_jobs=13)(delayed(cal_ctr_fun)(i,full_dl,df) for i in tqdm(range(len(full_dl) - window + 1)))
    ctr = pd.concat(ctr).reset_index(drop=True)
    # feather.write_dataframe(ctr, os.path.join(save_path, 'ctr.feather'))

    # jump_ctr因子计算----------------------------------------------------------------------------------------------
    print('jump_ctr因子计算中：')
    jump_ctr = Parallel(n_jobs=13)(delayed(cal_jump_ctr_fun)(i,full_dl,df) for i in tqdm(range(len(full_dl) - window + 1)))
    jump_ctr = pd.concat(jump_ctr).reset_index(drop=True)
    # feather.write_dataframe(jump_ctr, os.path.join(save_path, 'jump_ctr.feather'))
    return df[['TICKER', 'DATE', 'OvernightSmart20']],ctr,jump_ctr
    # return df[['TICKER', 'DATE', 'OvernightSmart20']]

if __name__ == '__main__':
    # 拉取数据：
    save_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\2021.6-2025.8'
    save_path1=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\2020-2022.12'
    flag=True
    # --------------------------------------------------------------
    # test:
    # if os.path.exists(os.path.join(save_path, 'test.feather')):
    #     old_df = feather.read_dataframe(os.path.join(save_path, 'test.feather'))
    #     old_last_day = max(old_df.DATE)
    #     old_time_list = old_df.DATE.drop_duplicates().to_list()
    #     sorted(old_time_list)
    #     df, full_dl = read_data(time_start='20210101', time_end='20230105')
    #     OvernightSmart20 = cal(df, full_dl)
    #     for file, data in {'OvernightSmart20.feather': OvernightSmart20}.items():
    #         old_df = feather.read_dataframe(os.path.join(save_path, 'test.feather'))
    #         new_df = data[data.DATE > old_df.DATE.max()]
    #         update_df = pd.concat([old_df, new_df]).reset_index(drop=True).drop_duplicates()
    #         feather.write_dataframe(update_df, os.path.join(save_path, 'test.feather'))
    #         update_df.to_csv(os.path.join(save_path, 'test2.csv'))
    #     flag = False
    # else:
    #     df, full_dl = read_data(time_start='20200101', time_end='20221231')
    #     OvernightSmart20 = cal(df, full_dl)
    #     for file, data in {'OvernightSmart20.feather': OvernightSmart20}.items():
    #         # feather.write_dataframe(data, os.path.join(save_path1, file))
    #         feather.write_dataframe(data, os.path.join(save_path, 'test.feather'))
    #         data.to_csv(os.path.join(save_path, 'test1.csv'))
    # ---------------------------------------------------------------
    while flag:
        if os.path.exists(os.path.join(save_path,'ctr.feather')):
            old_df=feather.read_dataframe(os.path.join(save_path,'ctr.feather'))
            old_last_day=max(old_df.DATE)
            old_time_list=old_df.DATE.drop_duplicates().to_list()
            sorted(old_time_list)
            # new_start=old_time_list[max(old_time_list.index(old_last_day)-20*3-5,0)]
            new_start=old_time_list[max(old_time_list.index(old_last_day)-100,0)]
            df, full_dl = read_data(time_start=new_start, time_end='20250807')
            OvernightSmart20,ctr,jump_ctr=cal(df, full_dl)
            for file,data in {'OvernightSmart20.feather':OvernightSmart20,'ctr.feather':ctr,'jump_ctr.feather':jump_ctr}.items():
                old_df = feather.read_dataframe(os.path.join(save_path, file))
                test_date=old_time_list[old_time_list.index(old_df.DATE.max())-15:]
                test_df=old_df[old_df.DATE.isin(test_date)].merge(data[data.DATE.isin(test_date)],on=['DATE','TICKER'])
                if np.isclose(test_df.iloc[:, 2], test_df.iloc[:, 3]).all():
                    new_df=data[data.DATE>old_df.DATE.max()]
                    update_df = pd.concat([old_df, new_df]).reset_index(drop=True).drop_duplicates().dropna()
                    feather.write_dataframe(update_df,os.path.join(save_path,file))
                else:
                    print('检查更新，数据出问题！')
                    exit()
            flag=False
        else:
            df, full_dl = read_data(time_start='20200101', time_end='20221231')
            OvernightSmart20,ctr,jump_ctr=cal(df, full_dl)
            for file, data in {'OvernightSmart20.feather': OvernightSmart20, 'ctr.feather': ctr,'jump_ctr.feather': jump_ctr}.items():
                feather.write_dataframe(data.dropna(), os.path.join(save_path1, file))
                feather.write_dataframe(data.dropna(), os.path.join(save_path, file))