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

        self.daily_path = daily_path or r'Z:\local_data\Data_Storage\daily.feather'
        self.daily_df = feather.read_dataframe(self.daily_path)[['DATE', 'TICKER', 'close']]
        self.daily_df.sort_values(['TICKER', 'DATE'], inplace=True)
        self.daily_df['pre_close'] = self.daily_df.groupby('TICKER')['close'].shift(1)  # close数据用前一天的
        # self.daily_df['pre_close'] = self.daily_df['close']
        self.daily_df = self.daily_df[self.daily_df['DATE'] > '20191229'][['DATE', 'TICKER', 'pre_close']]

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
                dt = dt[(dt['tradetime'] >= 92000000) & (dt['tradetime'] < 92459950)].reset_index(drop=True)
                dt['active_flow_dir'] = np.where((dt['Side'] == '1') & (dt['当前撮合价格'] > dt['上一笔撮合价格']), 1,
                                  np.where((dt['Side'] == '2') & (dt['当前撮合价格'] < dt['上一笔撮合价格']),-1, np.nan))# 1代表主动流入，-1代表主动流出
                # -----------------------------------------------------------------------
                dt['amount'] = dt['Price'] * dt['TradeQty']
                dt['active_netflow_money'] = dt['amount'] * dt['active_flow_dir']
                dt['active_netflow_qty'] = dt['TradeQty'] * dt['active_flow_dir']
                dt = dt.groupby('TICKER').filter(lambda x: len(x) >= 3)
                dt['amount_ranks'] = dt.groupby('TICKER')['amount'].rank(method='first')
                dt['大小单'] = dt.groupby('TICKER')['amount_ranks'].transform(lambda x: pd.cut(x, bins=[x.min(),
                               np.percentile(x,50),np.percentile(x,75),x.max()],labels=['小', '中','大']))
                # print(dt)
                # -----------------------------------------------------------------------------
                dt['sec'] = dt['tradetime'] // 1000
                seclist = dt['sec'].unique().tolist()
                seclist.sort()
                stampmap = {}
                for x in range(0, len(seclist), self.split_sec):
                    for j in range(self.split_sec):
                        stampmap[seclist[min(x + j, len(seclist) - 1)]] = x // self.split_sec
                dt['time_range'] = dt['sec'].map(stampmap)
                dt = dt.sort_values(by=['TICKER', 'ApplSeqNum']).reset_index(drop=True)
                split_by_sec = dt.groupby(['TICKER', 'time_range']).agg({'当前撮合成交量': ['first', 'last'],
                   'Price': ['first', 'last', 'max', 'min'],'active_netflow_money': 'sum','active_netflow_qty': 'sum'}).reset_index()
                split_by_sec.columns = ['TICKER', 'time', 'vol_first', 'vol_last', 'open', 'close', 'high', 'low','主动净流入额', '主动净流入量']
                tmp = dt.groupby(['TICKER', 'time_range', '大小单']).agg({'amount':['sum','size']}).reset_index()
                tmp.columns=['TICKER','time','大小单','amount','size']
                tmp = tmp.pivot(index=['TICKER', 'time'], columns='大小单', values=['amount','size']).reset_index()
                tmp.columns=['TICKER','time','小单amt','中单amt','大单amt','小单笔数','中单笔数','大单笔数']
                split_by_sec = split_by_sec.merge(tmp, on=['TICKER', 'time'], how='inner')
                split_by_sec['volume_diff'] = split_by_sec['vol_last'] - split_by_sec['vol_first']
                split_by_sec['DATE'] = date
                split_by_sec.drop(columns=['vol_first', 'vol_last'], inplace=True)
                dt = split_by_sec.reset_index(drop=True)
                dt.rename(columns={'volume_diff': 'volume'}, inplace=True)
                # ----------------------------------------------------
                res0=dt.groupby('TICKER').agg({'主动净流入额':'mean'}).reset_index()
                res0.columns=['TICKER','主动净流入额_mean']
                # 选出最后10s/60s的数据，作为有效数据
                end_dt=dt.groupby('TICKER').tail(1)[['TICKER','close']]
                trade10=dt.groupby('TICKER').tail(2)
                trade60=dt.groupby('TICKER').tail(12)

                res1=trade10.groupby('TICKER').agg({'主动净流入额':['mean','std'],'主动净流入量':['mean','std'],'大单amt':['mean','std'],'大单笔数':['mean','std']})
                res1.columns=['最后10s'+col[0]+'_'+col[1] for col in res1.columns]
                res2=trade60.groupby('TICKER').agg({'主动净流入额':['mean','std'],'主动净流入量':['mean','std'],'大单amt':['mean','std'],'大单笔数':['mean','std']})
                res2.columns=['最后60s'+col[0]+'_'+col[1] for col in res2.columns]

                res=res1.merge(res2,right_index=True,left_index=True,how='inner').reset_index().merge(end_dt,on='TICKER',how='inner').merge(res0,on='TICKER',how='inner')
                # print(dt)
                res['DATE']=date
                return res
        # df = get_daily('20220128')
        # print(df)
        df = Parallel(n_jobs=10)(delayed(get_daily)(date) for date in tqdm(date_list))
        df = pd.concat(df).reset_index(drop=True)
        alls = pd.concat([df_old, df]).reset_index(drop=True)
        alls=alls.merge(self.daily_df,on=['TICKER','DATE'],how='left')
        alls['ret']=alls['close']/alls['pre_close']-1
        alls.drop(columns=['close','pre_close'],inplace=True)
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
        # self.cal()


if __name__ == '__main__':
    obj = Auc_fac(start='20220101', end='20241231',
                       df_path=r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time\concat_daily',
                       savepath=r'C:\Users\admin\Desktop',
                       split_sec=5)
    obj.run()