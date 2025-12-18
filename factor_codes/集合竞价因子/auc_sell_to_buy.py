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
                # dt['TradeQty'] = np.where(dt['type'] == '4', -1 * dt['TradeQty'], dt['TradeQty'])
                end_dt = dt.groupby('TICKER').tail(1)[['TICKER','当前撮合成交量']].set_index('TICKER')
                # end_dt['撮合成交额']=end_dt['当前撮合价格']*end_dt['当前撮合成交量']
                # end_dt=end_dt[['撮合成交额']]
                dt = dt.groupby('TICKER', as_index=False).apply(lambda x: x.head(-1)).reset_index(drop=True)
                # dt['amount']=dt['Price']*dt['TradeQty']
                dt2=dt[dt['tradetime']>=92000000]
                dt3=dt[dt['tradetime']>=92454000]
                # ===========================================================
                # 因子二：用原因子里的sell方-buy方
                time2_buy = dt2[dt2['Side'] == '1'].groupby('TICKER').agg({'TradeQty': 'sum'}).rename(columns={'TradeQty': 'time2_buyQty'})
                time2_sell = dt2[dt2['Side'] == '2'].groupby('TICKER').agg({'TradeQty': 'sum'}).rename(columns={'TradeQty': 'time2_sellQty'})
                res2=time2_buy.merge(time2_sell, right_index=True,left_index=True, how='outer')
                res2.replace(np.nan,0,inplace=True)
                res2['NetSellQty']=res2['time2_sellQty']/res2['time2_buyQty']
                res2=res2[['NetSellQty']]#.reset_index()
                # ============================================================
                # 因子三：用原因子里的sell方-buy方，时间锁定在差不多最后撮合成交5s内
                time3_buy = dt3[dt3['Side'] == '1'].groupby('TICKER').agg({'TradeQty': 'sum'}).rename(columns={'TradeQty': 'time3_buyQty'})
                time3_sell = dt3[dt3['Side'] == '2'].groupby('TICKER').agg({'TradeQty': 'sum'}).rename(columns={'TradeQty': 'time3_sellQty'})
                res3 = time3_buy.merge(time3_sell, right_index=True, left_index=True, how='outer')
                # res3.replace(np.nan, 0, inplace=True)
                res3['NetSellQty_last5s'] = res3['time3_sellQty']/res3['time3_buyQty']
                res3 = res3[['NetSellQty_last5s']]#.reset_index()
                res=res2.merge(res3,right_index=True,left_index=True,how='outer')
                res.reset_index(inplace=True)
                res['DATE']=date
                return res
        # df = get_daily('20220104')
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