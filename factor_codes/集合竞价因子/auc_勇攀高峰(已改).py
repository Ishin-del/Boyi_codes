import os
import feather
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib.parallel import Parallel, delayed

class Climb_High:

    def __init__(self, start, end, calender_path=None, df_path=None, savepath=None, split_sec=1):
        self.start = start or '20220104'
        self.end = end or '20251121'
        self.df_path = df_path or r'D:\zkh\Price_and_vols_at_auction_time\concat_daily'
        self.calender_path = calender_path or r'\\192.168.1.210\Data_Storage2'
        self.savepath = savepath or rf'D:\因子保存\{self.__class__.__name__}'
        # self.tmp_path = os.path.join(self.savepath, 'basic')

        os.makedirs(self.savepath, exist_ok=True)
        # os.makedirs(str(self.tmp_path), exist_ok=True)
        self.split_sec = split_sec
        calendar = pd.read_csv(os.path.join(self.calender_path, 'calendar.csv'), dtype={'trade_date': str})
        calendar.rename(columns={'trade_date': 'DATE'}, inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']

    def __daily_calculation(self):
        if os.path.exists(os.path.join(self.savepath, 'auc_cov_pure.feather')):
            df_old = feather.read_dataframe(os.path.join(self.savepath, 'auc_cov_pure.feather'))
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
                warnings.filterwarnings('ignore')
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
                # --------------------------------------------------------------------------------
                dt[['open', 'close', 'high', 'low']] = dt[['open', 'close', 'high', 'low']].replace(0,np.nan)
                dt.sort_values(by=['TICKER','time']).reset_index(drop=True)
                for c in ['open', 'close', 'high', 'low']:
                    dt[c] = dt.groupby('TICKER')[c].ffill()
                for col in ['open', 'close', 'high', 'low']:
                    for i in [1, 2, 3,4]:
                        dt[f'{col}_{i}'] = dt.groupby('TICKER')[col].shift(i)
                use_col = ['open', 'close', 'high', 'low','open_1', 'close_1', 'high_1', 'low_1','open_2', 'close_2', 
                        'high_2', 'low_2','open_3', 'close_3', 'high_3', 'low_3','open_4', 'close_4', 'high_4', 'low_4']
                dt['last_5_std'] = dt[use_col].std(axis = 1, skipna = False)
                dt['last_5_mean'] = dt[use_col].mean(axis = 1, skipna = False)
                dt['advanced_volatility'] = (dt['last_5_std'] / dt['last_5_mean']) ** 2
                dt['ret'] = dt['close'] / dt['close_1'] # 收益波动比
                dt['ret_pure'] = dt['ret']- 1
                dt['ret_pure_volatility'] = dt['ret_pure'] / dt['advanced_volatility']
                dt['ad_vola_mean'] = dt.groupby('TICKER')['advanced_volatility'].transform('mean')
                dt['ad_vola_std'] = dt.groupby('TICKER')['advanced_volatility'].transform('std')
                dt = dt[dt['advanced_volatility'] > (dt['ad_vola_mean'] + dt['ad_vola_std'])]
                cov_pure_factor = dt.groupby('TICKER').apply(lambda x:x[['advanced_volatility','ret_pure_volatility']].cov().iloc[0,1])
                if cov_pure_factor.empty:
                    return
                result_df = cov_pure_factor.rename('cov_pure')
                result_df = result_df.reset_index().rename(columns = {'index':'TICKER'})
                result_df['DATE'] = date
                return result_df
        df = Parallel(n_jobs=10)(delayed(get_daily)(date) for date in tqdm(date_list))
        df = pd.concat(df).reset_index(drop=True)
        df.rename(columns={'cov_pure': 'auc_cov_pure'}, inplace=True)
        alls = pd.concat([df_old, df])
        alls = alls.drop_duplicates(subset=['TICKER', 'DATE'], keep='first').reset_index(drop=True)
        alls.replace([np.inf, -np.inf], np.nan, inplace=True)
        alls['auc_cov_pure']=alls['auc_cov_pure'].fillna(alls.groupby('DATE')['auc_cov_pure'].transform('median'))
        feather.write_dataframe(alls, os.path.join(self.savepath, 'auc_cov_pure.feather'))
        # print(alls)

    def run(self):
        self.__daily_calculation()
    '''
    剩下的就是对'cov_pure'按天做roll.mean, roll.std
    '''

if __name__ == '__main__':
    obj=Climb_High(start='20220104',end='20220301',
                   df_path=r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time\concat_daily',
                   savepath=r'C:\Users\admin\Desktop',
                   split_sec=5)
    obj.run()

