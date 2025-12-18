# -*- coding: utf-8 -*-
# Created on  2024/6/11 11:22
# @Author: Yangyu Che
# @File: 一视同仁.py
# @Contact: cheyangyu@126.com
# @Software: PyCharm

"""
《成交量激增与骤降时刻的对称性与“一视同仁”因子构建——多因子选股系列研究之十八》
"""

import os
import feather
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import stats
"""
这个因子可以考虑两个版本
1. 完全使用正成交量时刻
2. 使用正成交量时刻判断激增和骤降后，用全体时刻凑成“五分钟”
"""

class No_Exception:
    def __init__(self, start, end, calender_path=None, df_path=None, savepath=None,split_sec=1):
        self.start = start or '20220104'
        self.end = end or '20251121'
        self.df_path = df_path or r'D:\zkh\Price_and_vols_at_auction_time\concat_daily'
        self.calender_path = calender_path or r'\\192.168.1.210\Data_Storage2'
        self.savepath = savepath or rf'D:\因子保存\{self.__class__.__name__}'
        self.tmp_path = os.path.join(self.savepath, 'basic')

        os.makedirs(self.savepath, exist_ok=True)
        os.makedirs(str(self.tmp_path), exist_ok=True)
        # self.daily_list_file = os.listdir(r'D:\zkh\Price_and_vols_at_auction_time\concat_daily')
        self.split_sec = split_sec
        calendar = pd.read_csv(os.path.join(self.calender_path, 'calendar.csv'), dtype={'trade_date': str})
        calendar.rename(columns={'trade_date': 'DATE'}, inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']

    def __daily_calculation(self):
        if os.path.exists(os.path.join(self.tmp_path, '一视同仁.feather')):
            df_old = feather.read_dataframe(os.path.join(self.tmp_path, '一视同仁.feather'))
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []
        max_min_date = max(os.listdir(self.df_path)).replace('.feather', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)

        def get_date(df):
            df = df[df['volume'] > 0]
            if df.shape[0] <= 5:
                return None
            df['return_std'] = df['return'].iloc[::-1].rolling(5).std().iloc[::-1]
            boxcox_vol = pd.Series(stats.boxcox(df['volume'], lmbda=None)[0], index=df.index, name='vol_boxcox')
            boxcox_vol_diff = boxcox_vol - boxcox_vol.shift(1)
            boxcox_mean_std = boxcox_vol_diff.mean() + boxcox_vol_diff.std()

            bright_std = df[boxcox_vol_diff > boxcox_mean_std]['return_std'].mean()
            dark_std = df[boxcox_vol_diff < boxcox_mean_std]['return_std'].mean()

            bright_rtn = df[boxcox_vol_diff > boxcox_mean_std]['return'].mean()
            dark_rtn = df[boxcox_vol_diff < boxcox_mean_std]['return'].mean()

            std_fairness = bright_std - dark_std
            rtn_fairness = bright_rtn - dark_rtn
            return pd.Series([std_fairness, rtn_fairness], index=['std_fairness', 'rtn_fairness'])

        def get_daily(date):
            """
            每日分钟数据计算因子的函数
            单独定义出来 方便多进程调用
            """
            warnings.filterwarnings(action='ignore')
            filename = date + '.feather'
            if not os.path.exists(os.path.join(self.df_path, filename)):
                if date <= max_min_date:
                    raise FileNotFoundError
            else:
                dt = feather.read_dataframe(os.path.join(self.df_path, f'{date}.feather'))
                dt = dt[(dt['tradetime'] >= 92000000) & (dt['tradetime'] < 92459950)]
                dt['sec'] = dt['tradetime'] // 1000
                seclist = dt['sec'].unique().tolist()
                seclist.sort()
                stampmap = {}
                for x in range(0, len(seclist), self.split_sec):
                    for j in range(self.split_sec):
                        stampmap[seclist[min(x + j, len(seclist) - 1)]] = x // self.split_sec
                dt['time_range'] = dt['sec'].map(stampmap)
                dt = dt.sort_values(by=['TICKER', 'ApplSeqNum']).reset_index(drop=True)
                split_by_sec = dt.groupby(['TICKER', 'time_range']).agg(
                    {'当前撮合成交量': ['first', 'last'], 'Price': ['first', 'last', 'max', 'min']}).reset_index()
                split_by_sec.columns = ['TICKER', 'time', 'vol_first', 'vol_last', 'open', 'close', 'high', 'low']
                split_by_sec['volume_diff'] = split_by_sec['vol_last'] - split_by_sec['vol_first']
                split_by_sec['DATE'] = date
                split_by_sec.drop(columns=['vol_first', 'vol_last'], inplace=True)
                dt = split_by_sec.reset_index(drop=True)
                dt.rename(columns={'volume_diff': 'volume'}, inplace=True)
                # ----------------------------------------------------------------------------------------
                dt.sort_values(['TICKER','time'],inplace=True)
                dt['return'] = dt.groupby('TICKER')['close'].pct_change() + 1
                dt['return'] = dt['return'].replace([np.inf,-np.inf],np.nan)

                inday_return = dt.groupby('TICKER').apply(lambda x: x['close'].iloc[-1] / x['open'].iloc[0])
                inday_return = inday_return.replace([np.inf, -np.inf], np.nan)
                fairness_weight_value = dt.groupby('TICKER').apply(get_date)
                std_fairness = fairness_weight_value['std_fairness'].abs() * inday_return
                rtn_fairness = fairness_weight_value['rtn_fairness'].abs() * inday_return
                result_df = pd.concat([std_fairness.rename('波动公平因子'), rtn_fairness.rename('收益公平因子')], axis=1,
                                      join='inner')
                result_df = result_df.reset_index().rename(columns={'index': 'TICKER'})
                result_df['DATE'] = date
                return result_df

        df = Parallel(n_jobs=10)(delayed(get_daily)(date) for date in tqdm(date_list))
        df = pd.concat(df).reset_index(drop=True)
        alls = pd.concat([df_old, df])
        alls = alls.drop_duplicates(subset=['TICKER', 'DATE'], keep='first').reset_index(drop=True)
        feather.write_dataframe(alls, os.path.join(self.tmp_path, '一视同仁.feather'))


    def cal(self):
        warnings.filterwarnings('ignore')
        df = feather.read_dataframe(os.path.join(self.tmp_path, '一视同仁.feather'))
        tar_col = list(np.setdiff1d(df.columns, ['TICKER', 'DATE']))
        for c in tar_col:
            tmp = df[['TICKER', 'DATE', c]]
            tmp.rename(columns={c: 'auc_' + c}, inplace=True)
            c = 'auc_' + c
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp[c] = tmp[c].fillna(tmp.groupby('DATE')[c].transform('median'))
            feather.write_dataframe(tmp, os.path.join(self.savepath, c + '.feather'))
            # print(tmp)

    def run(self):
        self.__daily_calculation()
        self.cal()


if __name__ == '__main__':
    obj=No_Exception(start='20220108',end='20220201',
        df_path=r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time\concat_daily',
                       savepath=r'C:\Users\admin\Desktop',
                       split_sec=5)
    obj.run()
