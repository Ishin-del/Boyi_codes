# @Author: Yixin Tian
# @File: risk_uncer.py
# @Date: 2025/9/24 10:22
# @Software: PyCharm

"""
东北证券：基于高频数据的风险不确定性因子
"""
import feather
import numpy as np
import warnings
import pandas as pd
import os
from joblib import Parallel, delayed
from tqdm import tqdm


class Risk_uncer:
    def __init__(self, start, end, calender_path=None, df_path=None, savepath=None, wind_A_path=None, split_sec=1):
        self.start = start or '20220104'
        self.end = end or '20251121'
        self.df_path = df_path or r'D:\zkh\Price_and_vols_at_auction_time\concat_daily'
        self.calender_path = calender_path or r'\\192.168.1.210\Data_Storage2'
        self.savepath = savepath or rf'D:\因子保存\{self.__class__.__name__}'
        self.tmp_path = os.path.join(self.savepath, 'basic')
        self.wind_A_path=wind_A_path or r'Z:\local_data\Data_Storage\881001.WI.csv'

        os.makedirs(self.savepath, exist_ok=True)
        os.makedirs(str(self.tmp_path), exist_ok=True)
        # self.daily_list_file = os.listdir(r'D:\zkh\Price_and_vols_at_auction_time\concat_daily')
        self.split_sec = split_sec
        calendar = pd.read_csv(os.path.join(self.calender_path, 'calendar.csv'), dtype={'trade_date': str})
        calendar.rename(columns={'trade_date': 'DATE'}, inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']

    def __daily_calculation(self):
        if os.path.exists(os.path.join(self.tmp_path, 'auc_risk_uncer.feather')):
            df_old = feather.read_dataframe(os.path.join(self.tmp_path, 'auc_risk_uncer.feather'))
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []
        max_min_date = max(os.listdir(self.df_path)).replace('.feather', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)

        def get_daily(date):
            warnings.filterwarnings('ignore')
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
                # -----------------------------------------
                dt['ret']=dt.groupby('TICKER')['close'].pct_change(periods=4)
                dt.replace([np.inf, -np.inf], np.nan, inplace=True)
                dt['log_ret'] = np.log(1 + dt['ret'])
                dt['log_ret_square'] = dt['log_ret'] ** 2
                dt.replace([np.inf, -np.inf], np.nan, inplace=True)
                dt['VaR'] = dt.groupby('TICKER')['ret'].transform(lambda x: x.quantile(0.95))
                dt['VaR_RT'] = dt.groupby('TICKER')['ret'].transform(lambda x: x.quantile(1 - 0.95))
                # -----------------------------------------
                dt=dt.groupby('TICKER').apply(get_date).reset_index()
                dt[['DATE','RV','cVaR', 'cVaR_RT']]=dt[0].apply(pd.Series)
                dt.drop(columns=0,inplace=True)
                return dt

        def get_date(x):
            date=x['DATE'].iloc[0]
            RV=x['log_ret_square'].sum()
            cVaR=-x[x['ret']<=x['VaR']]['ret'].mean()
            cVaR_RT=-x[x['ret']>=x['VaR_RT']]['ret'].mean()
            return [date,RV,cVaR, cVaR_RT]

        df = Parallel(n_jobs=10)(delayed(get_daily)(date) for date in tqdm(date_list))
        df = pd.concat(df).reset_index(drop=True)
        alls = pd.concat([df_old, df]).reset_index(drop=True)
        feather.write_dataframe(alls, os.path.join(self.tmp_path, 'auc_risk_uncer.feather'))

    def cal(self):
        warnings.filterwarnings('ignore')
        df = feather.read_dataframe(os.path.join(self.tmp_path, 'auc_risk_uncer.feather'))
        tar_col = list(np.setdiff1d(df.columns, ['TICKER', 'DATE']))
        for c in tar_col:
            tmp = df[['TICKER', 'DATE', c]]
            tmp.rename(columns={c: 'auc_' + c}, inplace=True)
            c = 'auc_' + c
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp[c] = tmp[c].fillna(tmp.groupby('DATE')[c].transform('median'))
            # print(tmp)
            feather.write_dataframe(tmp, os.path.join(self.savepath, c + '.feather'))

    def run(self):
        self.__daily_calculation()
        self.cal()

"""
跳过的都是这些列按天 roll.mean
"""

if __name__=='__main__':
    obj=Risk_uncer(start='20220107',end='20220115',
        df_path=r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time\concat_daily',
                       savepath=r'C:\Users\admin\Desktop',
                       split_sec=5)
    obj.run()