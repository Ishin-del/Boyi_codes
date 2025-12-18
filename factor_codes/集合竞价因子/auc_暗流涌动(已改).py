import feather
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from tool_tyx.path_data import DataPath
import os
import warnings
from tqdm import tqdm

from tool_tyx.tyx_funcs import process_na_stock

class 暗流涌动:
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
        # todo:
        if os.path.exists(os.path.join(self.tmp_path, '暗流涌动.feather')):
            df_old = feather.read_dataframe(os.path.join(self.tmp_path, '暗流涌动.feather'))
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []
        max_min_date = max(os.listdir(self.df_path)).replace('.feather', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)

        def get_date(x):
            warnings.filterwarnings('ignore')
            # 于股票日内交易量的分布特征,香农熵（Shannon Entropy）刻画，熵越大，个股日内相对交易量的分布越趋于均匀分布，否则，股票的交易越有可能由信息驱动
            x['vol_min_rela_sum']=x['rela_min_vol'].sum()
            x['total_rela_vol_5min']=x.groupby('time_group')['rela_min_vol'].transform('sum')
            x['ratio_5min']=x['total_rela_vol_5min']/x['vol_min_rela_sum'] #占比p(xi)
            tmp1=x[['time_group', 'ratio_5min']].drop_duplicates()
            entropy = -np.sum(tmp1['ratio_5min'] * np.log2(tmp1['ratio_5min']))
            return entropy

        def get_daily(date):
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
                dt=dt[['TICKER', 'time', 'open', 'high', 'low', 'close', 'volume','DATE']]
                # -----------------------------------------------------
                dt.sort_values(['TICKER', 'time'], inplace=True)
                dt['time_group']=dt.groupby('TICKER').cumcount() // 5
                dt['vol_total_time']=dt.groupby('time')['volume'].transform('sum') # 同时刻下，全市场所有股票成交量
                dt['rela_min_vol'] = dt['volume'] / dt['vol_total_time']  # 同时刻个股的相对成交量（个股成交量/全市场所有股票成交量
                dt['price_volatility'] = (dt['high'] - dt['low']) / dt['open']
                entropy=dt.groupby('TICKER').apply(get_date).reset_index().rename(columns={0:'entropy'})
                entropy['entropy_adjust'] = np.abs(entropy['entropy'] - entropy['entropy'].mean())
                # -----------------------
                """
                下面这部分没加到get_date里，感觉加进去还不如直接这样写方便，如果你需要我加到get_date里，跟我说一声
                """
                dt['vol_mean_5min'] = dt.groupby('TICKER').rolling(5)['volume'].mean().values
                dt['price_volatility'] = (dt['high'] - dt['low']) / dt['open']
                dt['flag'] = np.where(dt['volume'] > 2 * dt['vol_mean_5min'], 'rush_mean','normal_mean')  # 1:激增时刻,0:普通时刻
                tmp = dt.groupby(['TICKER', 'flag'])['price_volatility'].mean().reset_index()
                tmp = tmp.pivot(index='TICKER', columns='flag',values='price_volatility').reset_index()  # .rename(columns={0:'normal_mean',1:'rush_mean'})
                tmp['elasticity'] = 1 - tmp['rush_mean'] / tmp['normal_mean']
                tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
                tmp['elasticity_adjust'] = np.abs(tmp['elasticity'] - tmp['elasticity'].mean())
                dt=entropy[['TICKER','entropy_adjust']].merge(tmp[['TICKER','elasticity_adjust']],on='TICKER',how='inner')
                dt['DATE']=date
                return dt

        df = Parallel(n_jobs=10)(delayed(get_daily)(date) for date in tqdm(date_list))
        df = pd.concat(df).reset_index(drop=True)
        """
        原因子，根据df合成之后，根据日期roll20算mean和std，最后合成暗流涌动因子。所以我代码就先写到这了，需要改，望告知
        """
        alls = pd.concat([df_old, df])
        alls = alls.drop_duplicates(subset=['TICKER', 'DATE'], keep='first').reset_index(drop=True)
        feather.write_dataframe(alls, os.path.join(self.tmp_path, '暗流涌动.feather'))

    def cal(self):
        warnings.filterwarnings('ignore')
        df = feather.read_dataframe(os.path.join(self.tmp_path, '暗流涌动.feather'))
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

if __name__=='__main__':
    obj=暗流涌动(start='20220111',end='20220120',
                df_path=r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time\concat_daily',
                savepath=r'C:\Users\admin\Desktop',
                split_sec=5)
    obj.run()