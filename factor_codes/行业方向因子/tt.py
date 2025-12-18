import warnings

import feather
import os
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from tqdm import tqdm


class Sector_fac:
    def __init__(self, start, end, calender_path=None, df_path=None, savepath=None, citi_path=None,daily_path=None,split_sec=1):
        self.start = start or '20220104'
        self.end = end or '20251121'
        self.df_path = df_path or r'D:\zkh\Price_and_vols_at_auction_time\concat_daily'
        self.calender_path = calender_path or r'\\192.168.1.210\Data_Storage2'
        self.savepath = savepath or rf'D:\因子保存\{self.__class__.__name__}'
        self.tmp_path = os.path.join(self.savepath, 'basic')

        os.makedirs(self.savepath, exist_ok=True)
        os.makedirs(str(self.tmp_path), exist_ok=True)

        self.split_sec = split_sec
        calendar = pd.read_csv(os.path.join(self.calender_path, 'calendar.csv'), dtype={'trade_date': str})
        calendar.rename(columns={'trade_date': 'DATE'}, inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']

        self.citi_path=r'Z:\local_data\Data_Storage\citic_code.feather' or citi_path
        self.citi_df=feather.read_dataframe(self.citi_path).drop(columns='CITIC_NAME')
        # self.citi_df['CITIC_CODE_sec']=self.citi_df['CITIC_CODE'].str[:7]
        self.citi_df['CITIC_CODE']=self.citi_df['CITIC_CODE'].str[:7]
        self.citi_df.sort_values(['TICKER', 'DATE'], inplace=True)
        self.citi_df['CITIC_CODE'] = self.citi_df.groupby('TICKER')['CITIC_CODE'].shift(1)

        self.daily_path = daily_path or r'Z:\local_data\Data_Storage\daily.feather'
        self.daily_df = feather.read_dataframe(self.daily_path)[['DATE', 'TICKER', 'close']]
        self.daily_df.sort_values(['TICKER', 'DATE'], inplace=True)
        self.daily_df['pre_close'] = self.daily_df.groupby('TICKER')['close'].shift(1)  # close数据用前一天的
        self.daily_df.drop(columns='close',inplace=True)


    def __daily_calculation(self):
        if os.path.exists(os.path.join(self.tmp_path, 'sec_code.feather')):
            df_old = feather.read_dataframe(os.path.join(self.tmp_path, 'sec_code.feather'))
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []
        max_min_date = max(os.listdir(self.df_path)).replace('.feather', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)

        def get_daily(date):
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
                split_by_sec.columns = ['TICKER','vol_first', 'vol_last', 'open', 'close', 'high', 'low']
                split_by_sec['volume_diff'] = split_by_sec['vol_last'] - split_by_sec['vol_first']
                split_by_sec['DATE'] = date
                split_by_sec.drop(columns=['vol_first', 'vol_last'], inplace=True)
                dt = split_by_sec.reset_index(drop=True)
                dt.rename(columns={'volume_diff': 'volume'}, inplace=True)
                return dt
        # --------------------------------------------
        # dt = Parallel(n_jobs=12)(delayed(get_daily)(date) for date in tqdm(date_list))
        # dt = pd.concat(dt).reset_index(drop=True)
        # dt=get_daily('20220104')
        dt=[]
        # for date in tqdm(date_list):
        #     tmp=get_daily(date)
        #     dt.append(tmp)
        # dt = pd.concat(dt).reset_index(drop=True)
        # feather.write_dataframe(dt,r'C:\Users\admin\Desktop\集合竞价close.feather')
        dt=feather.read_dataframe(r'C:\Users\admin\Desktop\集合竞价close.feather')
        # ------------------------------------------------
        dt = dt.merge(self.daily_df, on=['TICKER', 'DATE'], how='inner')
        dt.sort_values(['TICKER','DATE'], inplace=True)
        dt = dt.groupby(['TICKER','DATE'])[['TICKER', 'DATE', 'pre_close', 'close']].tail(1)
        # dt['auc_ret']=dt['close']/dt['pre_close']  #-1
        dt['auc_ret'] = dt['pre_close'] / dt['close']  # -1
        dt = dt[['DATE', 'TICKER', 'auc_ret']]

        dt = dt.merge(self.citi_df, on=['DATE', 'TICKER'], how='inner')
        dt['code_percent'] = dt.groupby(['CITIC_CODE','DATE'])['auc_ret'].rank(pct=True, method='average')  # 股票/行业
        dt['code_percent_mkt'] = dt.groupby('DATE')['auc_ret'].rank(pct=True, method='average')  # 股票/全市场
        tmp = dt[['CITIC_CODE','DATE','auc_ret']].groupby(['CITIC_CODE','DATE'])['auc_ret'].mean().reset_index().rename(
            columns={'auc_ret': 'sec_ret'})
        tmp['sec_percent'] = tmp.groupby('DATE')['sec_ret'].rank(pct=True, method='average')  # 行业/全市场行业
        dt = dt.merge(tmp, on=['CITIC_CODE','DATE'], how='inner')

        # todo:test
        dt['tt'] = dt['code_percent_mkt'] - dt['sec_percent'] + dt['code_percent']
        dt['factor_2'] = (1 - dt['sec_percent']) * dt['code_percent']
        dt['factor_6'] = (dt['sec_percent'] + dt['code_percent']) / dt['code_percent_mkt']
        dt['factor_9'] = dt['code_percent_mkt'] / dt['sec_percent']
        dt['tt2'] = dt['factor_2'] + dt['factor_6'] + dt['factor_9']
        dt = dt.drop(columns=['auc_ret', 'CITIC_CODE', 'code_percent', 'code_percent_mkt',
                              'sec_ret', 'sec_percent'])
        dt.replace([np.inf, -np.inf], np.nan, inplace=True)
        alls = pd.concat([df_old, dt]).reset_index(drop=True)
        alls=alls[~alls['DATE'].isin(['20240206', '20240207', '20240208', '20240926', '20240927',
                                      '20240930', '20241008'])]
        feather.write_dataframe(alls, os.path.join(self.tmp_path, 'sec_code.feather'))
        print(alls)

    def cal(self):
        warnings.filterwarnings('ignore')
        data = feather.read_dataframe(os.path.join(self.tmp_path, 'sec_code.feather'))
        rescols = list(np.setdiff1d(data.columns, ['TICKER', 'DATE']))
        for c in tqdm(rescols):
            tmp = data[['TICKER', 'DATE', c]]
            tmp.columns=['TICKER', 'DATE', 'auc_' + c]
            c = 'auc_' + c
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp[c] = tmp[c].fillna(tmp.groupby('DATE')[c].transform('median'))
            feather.write_dataframe(tmp, os.path.join(self.savepath, c + '.feather'))
            print(tmp)

    def run(self):
        self.__daily_calculation()
        self.cal()


if __name__=='__main__':
    obj=Sector_fac(start='20220101',end='20241231',
        df_path=r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time\concat_daily',
                       savepath=r'C:\Users\admin\Desktop')
    obj.run()


