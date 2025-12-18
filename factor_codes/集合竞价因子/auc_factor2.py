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


class Auc_fac:
    warnings.filterwarnings('ignore')

    def __init__(self, start, end, calender_path=None, df_path=None, savepath=None, daily_path=None, citi_path=None,
                 split_sec=1):
        self.start = start or '20220104'
        self.end = end or '20251121'
        self.df_path = df_path or r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time_SZ_early\concat_daily'
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
        if os.path.exists(os.path.join(self.tmp_path, 'acu.feather')):
            df_old = feather.read_dataframe(os.path.join(self.tmp_path, 'acu.feather'))
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
                # dt = dt[(dt['tradetime'] >= 92000000) & (dt['tradetime'] < 92459950)].reset_index(drop=True)
                dt = dt[dt['tradetime'] < 92459950].reset_index(drop=True)
                dt.sort_values(['TICKER','ApplSeqNum'],inplace=True)
                end_dt=dt.groupby('TICKER').tail(1)[['TICKER','当前撮合成交量']].set_index('TICKER')
                dt=dt.groupby('TICKER',as_index=False).apply(lambda x:x.head(-1)).reset_index(drop=True)
                dt1=dt[dt['tradetime']< 91959999]
                dt1['TradeQty']=np.where(dt1['type']=='4',-1*dt1['TradeQty'],dt1['TradeQty'])
                dt2=dt[dt['tradetime']>=92000000]
                # ------------------------------
                time1_buy=dt1[dt1['Side']=='1'].groupby('TICKER').agg({'TradeQty':'sum'}).rename(columns={'TradeQty':'time1_buyQty'})
                time1_sell=dt1[dt1['Side']=='2'].groupby('TICKER').agg({'TradeQty':'sum'}).rename(columns={'TradeQty':'time1_sellQty'})
                time2_buy=dt2[dt2['Side']=='1'].groupby('TICKER').agg({'TradeQty':'sum'}).rename(columns={'TradeQty':'time2_buyQty'})
                time2_sell=dt2[dt2['Side']=='2'].groupby('TICKER').agg({'TradeQty':'sum'}).rename(columns={'TradeQty':'time2_sellQty'})
                res=end_dt.merge(time1_buy,right_index=True,left_index=True,how='inner').merge(time1_sell,right_index=True,
                    left_index=True,how='inner').merge(time2_buy,right_index=True,left_index=True,how='inner').merge(time2_sell,
                    right_index=True,left_index=True,how='inner')
                tar_col=np.setdiff1d(res.columns,['当前撮合成交量'])
                for col in tar_col:
                    res[col+'_tradeQty']=res['当前撮合成交量']/res[col]
                res=res[[col+'_tradeQty' for col in tar_col]].reset_index()
                # print(res)
                res['DATE']=date
                return res
        # df = get_daily('20220128')
        # print(df)
        df = Parallel(n_jobs=10)(delayed(get_daily)(date) for date in tqdm(date_list))
        df = pd.concat(df).reset_index(drop=True)
        alls = pd.concat([df_old, df]).reset_index(drop=True)
        feather.write_dataframe(alls, os.path.join(self.tmp_path, 'tt.feather'))
        print(alls)

    def cal(self):
        warnings.filterwarnings('ignore')
        df = feather.read_dataframe(os.path.join(self.tmp_path, 'tt.feather'))
        tar_col = list(np.setdiff1d(df.columns, ['TICKER', 'DATE']))
        for c in tar_col:
            tmp = df[['TICKER', 'DATE', c]]
            tmp.rename(columns={c: 'auc_' + c}, inplace=True)
            c = 'auc_' + c
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp[c] = tmp[c].fillna(tmp.groupby('DATE')[c].transform('median'))
            feather.write_dataframe(tmp, os.path.join(self.savepath,'test' ,c + '.feather'))
            # print(tmp)

    def run(self):
        self.__daily_calculation()
        self.cal()


if __name__ == '__main__':
    obj = Auc_fac(start='20220101', end='20241231',
                       df_path=r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time\concat_daily',
                       savepath=r'C:\Users\admin\Desktop',
                       split_sec=5)
    obj.run()