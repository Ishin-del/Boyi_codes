# @Author: Yixin Tian
# @File: factor_min_data.py
# @Date: 2025/9/4 10:27
# @Software: PyCharm
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import read_min_data,process_na_stock
import os
import feather
from joblib import Parallel,delayed

class Auc_factor:
    def __init__(self,start,end,df_path,save_path=None,split_sec=1):
        self.start = start
        self.end = end
        self.df_path = df_path
        self.save_path = save_path
        self.split_sec=split_sec
        calendar = pd.read_csv(DataPath.to_path + '\\calendar.csv', dtype={'trade_date': str})
        calendar.rename(columns={'trade_date': 'DATE'}, inplace=True)
        self.tar_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']

    def __daily_calculation(self):
        # todo:
        if os.path.exists(self.save_path + 'high_freq_skew.feather'):
            df_old = feather.read_dataframe(self.save_path + 'high_freq_skew.feather')
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []
        max_min_date = max(os.listdir(self.df_path)).replace('.feather', '')
        date_list = np.setdiff1d(self.tar_list, exist_date)

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
                dt.rename(columns={'volume_diff':'volume'},inplace=True)
                # --------------------------------------
                dt = dt[['DATE', 'TICKER', 'time', 'volume', 'close']]
                dt.sort_values(['TICKER', 'time'], inplace=True)
                dt['ret_min'] = dt['close'].pct_change()
                dt.replace([np.inf, -np.inf], np.nan, inplace=True)
                dt.replace(np.nan, 0, inplace=True)
                dt=dt.groupby(['TICKER','DATE']).apply(get_date).reset_index().drop_duplicates(
                    subset=['TICKER','DATE'],keep='first').drop(columns='level_2')
            return dt

        def get_date(x):
            x['sum_cube']=sum(x['ret_min']**3)
            x['sum_square']=sum(x['ret_min']**2)
            x['df_len']=len(x['ret_min'])
            x['high_freq_skew']=x['ret_min'].skew()
            x['sum_square_low']=sum(x['ret_min'][x['ret_min']<0]**2)
            x['sum_square_up']=sum(x['ret_min'][x['ret_min']>0]**2)
            x['low_vola_ratio']=(np.sqrt(x['df_len']) * x['sum_square_low']) / x['sum_square']
            x['up_vola_ratio']=(np.sqrt(x['df_len']) * x['sum_square_up']) / x['sum_square']
            return x[['high_freq_skew','low_vola_ratio','up_vola_ratio']]

        df=Parallel(n_jobs=5)(delayed(get_daily)(date) for date in tqdm(date_list))
        if df:
            df=pd.concat(df).reset_index(drop=True)
        df=pd.concat([df_old,df]).sort_values(by=['TICKER','DATE']).reset_index(drop=True)
        return df

    def cal(self):
        warnings.filterwarnings('ignore')
        df=self.__daily_calculation()
        tar_col=np.setdiff1d(df.columns,['TICKER','DATE'])
        for c in tar_col:
            tmp = df[['TICKER', 'DATE', c]]
            tmp.rename(columns={c:'auc_'+c},inplace=True)
            c='auc_'+c
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp[c] = tmp[c].fillna(tmp.groupby('DATE')[c].transform('median'))
            feather.write_dataframe(tmp, os.path.join(self.save_path, c + '.feather'))
            # print(tmp)

    def run(self):
        self.cal()


if __name__=='__main__':
    obj=Auc_factor('20220104','20220204',
                   r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time\concat_daily',
                   r'C:\Users\admin\Desktop')
    obj.run()