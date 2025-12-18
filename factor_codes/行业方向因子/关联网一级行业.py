# @Author: Yixin Tian
# @File: traction_LUD.py
# @Date: 2025/9/28 14:53
# @Software: PyCharm

"""
开源证券：从涨跌停外溢行为到股票关联网络
"""

import os
import feather
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings


def process_na_stock(df, col):
    '''解决某些股票出现时间序列断开的情况，这种情况会在rolling（window!=min_period）的时候,因子值出现不同'''
    '''重组再上市情况'''
    pt = df.pivot_table(columns='TICKER', index='DATE', values=col)
    pt = pt.ffill()
    pp = pt.melt(ignore_index=False).reset_index(drop=False)
    df = pd.merge(pp, df, how='left', on=['TICKER', 'DATE'])
    use_cols = list(np.setdiff1d(df.columns, ['TICKER', 'DATE']))
    for c in use_cols:
        df[c] = df.groupby('TICKER')[c].ffill()
    df.drop(columns='value', inplace=True)
    return df


def process_na_auc(df, col):
    '''解决某些股票出现时间序列断开的情况，这种情况会在rolling（window!=min_period）的时候,因子值出现不同'''
    '''重组再上市情况'''
    pt = df.pivot_table(columns='TICKER', index='time', values=col)
    pt = pt.ffill()
    pp = pt.melt(ignore_index=False).reset_index(drop=False)
    df = pd.merge(pp, df, how='left', on=['TICKER', 'time'])
    use_cols = list(np.setdiff1d(df.columns, ['TICKER', 'time']))
    for c in use_cols:
        df[c] = df.groupby('TICKER')[c].ffill()
    df.drop(columns='value', inplace=True)
    return df


class Traction_LUD:
    warnings.filterwarnings('ignore')

    def __init__(self, start, end, calender_path=None, df_path=None, savepath=None, daily_path=None, citi_path=None,
                 split_sec=1):
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

        self.daily_path = daily_path or r'Z:\local_data\Data_Storage\daily.feather'
        self.daily_df = feather.read_dataframe(self.daily_path)[['DATE', 'TICKER', 'close']]
        self.daily_df.sort_values(['TICKER', 'DATE'], inplace=True)
        self.daily_df['close'] = self.daily_df.groupby('TICKER')['close'].shift(1)  # close数据用前一天的
        self.daily_df['pre_close'] = self.daily_df['close']
        self.daily_df = self.daily_df[self.daily_df['DATE'] > '20191229']
        self.daily_df.sort_values(['TICKER', 'DATE'], inplace=True)
        self.daily_df = process_na_stock(self.daily_df, 'close')
        self.daily_df['close_roll20'] = self.daily_df.groupby('TICKER')['close'].shift(20)
        self.daily_df['ret_roll20'] = self.daily_df['close'] / self.daily_df['close_roll20'] - 1
        self.daily_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # self.daily_df = self.daily_df[self.daily_df['DATE'].isin(self.daily_list)]
        # -----------------------------------------------------------------------------------
        self.citi_path = r'Z:\local_data\Data_Storage\citic_code.feather' or citi_path
        self.citi_df = feather.read_dataframe(self.citi_path).drop(columns='CITIC_NAME')
        # self.citi_df['CITIC_CODE_sec']=self.citi_df['CITIC_CODE'].str[:8]
        # self.citi_df['CITIC_CODE'] = self.citi_df['CITIC_CODE'].str[:7] #二级行业
        self.citi_df['CITIC_CODE'] = self.citi_df['CITIC_CODE'].str[:5] #一级行业
        self.citi_df.sort_values(['TICKER', 'DATE'], inplace=True)
        self.citi_df['CITIC_CODE'] = self.citi_df.groupby('TICKER')['CITIC_CODE'].shift(1)

    def __daily_calculation(self):
        if os.path.exists(os.path.join(self.tmp_path, 'traction_lud.feather')):
            df_old = feather.read_dataframe(os.path.join(self.tmp_path, 'traction_lud.feather'))
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []
        max_min_date = max(os.listdir(self.df_path)).replace('.feather', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)
        date_list = [x for x in date_list if x not in ['20240206', '20240207', '20240208', '20240926', '20240927',
                                                       '20240930', '20241008']]

        def get_daily(date):
            warnings.filterwarnings('ignore')
            filename = date + '.feather'
            if not os.path.exists(os.path.join(self.df_path, filename)):
                if date <= max_min_date:
                    raise FileNotFoundError
            else:
                dt = feather.read_dataframe(os.path.join(self.df_path, f'{date}.feather'))
                dt = dt[(dt['tradetime'] >= 92000000) & (dt['tradetime'] < 92459950)]
                dt = dt.sort_values(by=['TICKER', 'ApplSeqNum']).reset_index(drop=True)
                split_by_sec = dt.groupby(['TICKER']).agg(
                    {'当前撮合成交量': ['first', 'last'], 'Price': ['first', 'last', 'max', 'min']}).reset_index()
                split_by_sec.columns = ['TICKER', 'vol_first', 'vol_last', 'open', 'close', 'high', 'low']
                split_by_sec['volume_diff'] = split_by_sec['vol_last'] - split_by_sec['vol_first']
                split_by_sec['DATE'] = date
                split_by_sec.drop(columns=['vol_first', 'vol_last'], inplace=True)
                dt = split_by_sec.reset_index(drop=True)
                dt.rename(columns={'volume_diff': 'volume'}, inplace=True)
                # -------------------------------------------------------------
                dt = dt.merge(self.daily_df[['TICKER', 'DATE', 'pre_close']], on=['TICKER', 'DATE'], how='inner')
                # dt.sort_values(['TICKER', 'time'], inplace=True)
                dt = dt.groupby('TICKER')[['TICKER', 'DATE', 'pre_close', 'close']].tail(1)
                dt['auc_ret'] = dt['close'] / dt['pre_close']  # -1
                dt = dt[['DATE', 'TICKER', 'auc_ret']]

                dt = dt.merge(self.citi_df, on=['DATE', 'TICKER'], how='inner')
                dt['code_percent'] = dt.groupby('CITIC_CODE')['auc_ret'].rank(pct=True, method='average')  # 股票/行业
                dt['code_percent_mkt'] = dt['auc_ret'].rank(pct=True, method='average')  # 股票/全市场
                tmp = dt[['CITIC_CODE', 'auc_ret']].groupby('CITIC_CODE')['auc_ret'].mean().reset_index().rename(
                    columns={'auc_ret': 'sec_ret'})
                tmp['sec_percent'] = tmp['sec_ret'].rank(pct=True, method='average')  # 行业/全市场行业
                dt = dt.merge(tmp, on=['CITIC_CODE'], how='inner')
                # ==================================================
                dt['sec_code_percent'] = dt['sec_percent'] - dt['code_percent']
                dt['mkt_sec_diff'] = dt['code_percent_mkt'] - dt['sec_percent']
                dt['factor']=dt['code_percent']/dt['code_percent_mkt']
                dt['over_on_code'] = dt['code_percent_mkt'] - dt['sec_percent'] + dt['code_percent']

                pre_ret = dt[['TICKER', 'auc_ret']].drop_duplicates().reset_index(drop=True)
                ret_df3 = dt[['TICKER', 'factor']].drop_duplicates().reset_index(drop=True)
                # -------------------------------------------
                res = pd.crosstab(dt['TICKER'], dt['CITIC_CODE'])
                res = (res @ res.T).astype(int)
                result_df = (res > 0).astype(int)
                np.fill_diagonal(result_df.values, 0)
                # --------------------------------------
                tmp_values = result_df.apply(lambda x: x.values)
                tmp_codes = result_df.apply(lambda x: x.index)

                tmp_dict3 = ret_df3.set_index('TICKER')['factor'].to_dict()
                tmp_ret3 = tmp_codes.applymap(lambda x: tmp_dict3.get(x))
                res3 = (tmp_values * tmp_ret3).sum() / tmp_values.sum()
                res3 = pd.DataFrame(res3).reset_index()
                res3 = res3[['TICKER', 0]].rename(columns={0: 'exp_LUD3'})
                # ------------------------------------------------------
                res = res.merge(res3,on='TICKER',how='inner')
                res['DATE'] = date

                res = res.merge(pre_ret, on='TICKER', how='inner')
                res['traction_LUD'] = res['auc_ret'] - res['exp_LUD3']
                res.drop(columns='auc_ret', inplace=True)
                res=res[['TICKER','DATE','traction_LUD']]
                # .merge(dt[['TICKER', 'DATE', 'sec_code_percent', 'code_percent',
                #            'mkt_sec_diff', 'over_on_code']], on=['TICKER', 'DATE'], how='inner')
                return res

        # df=get_daily('20220104')
        # print(df)
        df = Parallel(n_jobs=10)(delayed(get_daily)(date) for date in tqdm(date_list))
        df = pd.concat(df).reset_index(drop=True)
        alls = pd.concat([df_old, df]).reset_index(drop=True)
        feather.write_dataframe(alls, os.path.join(self.tmp_path, 'traction_lud.feather'))
        print(alls)

    def cal(self):
        warnings.filterwarnings('ignore')
        df = feather.read_dataframe(os.path.join(self.tmp_path, 'traction_lud.feather'))
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
    obj = Traction_LUD(start='20220101', end='20241231',
                       df_path=r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time\concat_daily',
                       savepath=r'C:\Users\admin\Desktop',
                       split_sec=5)
    obj.run()