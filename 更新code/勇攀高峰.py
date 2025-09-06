# -*- coding: utf-8 -*-
# Created on  2023/6/7 11:22
# @Author: Yangyu Che
# @File: PATV.py
# @Contact: cheyangyu@126.com
# @Software: PyCharm


import os
import feather
import warnings
import numpy as np
import odbc
import pandas as pd
import datetime as dt
from tqdm import tqdm
from joblib.parallel import Parallel, delayed

from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import kill_mutil_process


class Climb_High:

    def __init__(self, start=None, end=None,output_path=None):
        self.start_date = start
        self.end_date = end

        self.data_path = DataPath.to_path
        # self.min_data_path = r'\\10.36.35.85\Data_storage\Min_Data'
        self.sz_min = DataPath.sz_min
        self.sh_min = DataPath.sh_min

        self.output_path = output_path
        self.tmp_path = DataPath.tmp_path
        self.calendar = pd.read_csv(os.path.join(self.data_path, 'calendar.csv'))['trade_date'].astype(str)
        self.daily = pd.DataFrame()

    def inner_day_volatility(self):
        date_list = self.calendar[(self.calendar >= self.start_date) & (self.calendar <= self.end_date)]
        output_path = os.path.join(self.tmp_path, 'inner_day_volatility.feather')
        if os.path.exists(output_path):
            df_old = feather.read_dataframe(output_path)
            exist_date = df_old['DATE'].unique()
        else:
            df_old = pd.DataFrame(columns=['TICKER', 'DATE'])
            exist_date = []

        # file_list = [x[:4] + '-' + x[4:6] + '-' + x[6:8] + '.feather' for x in np.setdiff1d(date_list, exist_date)]
        file_list = [x+ '.feather' for x in np.setdiff1d(date_list, exist_date)]

        if len(file_list) == 0:
            print('无需更新分钟因子')
            return

        def main_func(file):
            warnings.filterwarnings(action='ignore')
            pid = os.getpid()
            df_min_sz,df_min_sh=pd.DataFrame(),pd.DataFrame()
            try:
                df_min_sz = feather.read_dataframe(os.path.join(self.sz_min, file),columns=['TICKER', 'min', 'close', 'volume','open','high','low'])
            except FileNotFoundError:
                pass
            try:
                df_min_sh = feather.read_dataframe(os.path.join(self.sh_min, file),columns=['TICKER', 'min', 'close', 'volume','open','high','low'])
            except FileNotFoundError:
                pass
            df_min=pd.concat([df_min_sh,df_min_sz])
            if df_min.empty:
                return

            # date = df_min['trade_time'].iloc[0][:10].replace('-', '')
            df_min=df_min[df_min['min']>930]
            # start_time = df_min['trade_time'].min()
            # end_time = start_time[:11] + '14:57:00'
            start_time=1457
            end_time=1459
            df_min=df_min[(df_min['min']<start_time)]

            df_sig = df_min.groupby('TICKER').agg({'close': 'mean', 'volume': 'sum', 'min': len})
            valid_ts_codes = df_sig[(df_sig['close'] > 0) & (df_sig['volume'] > 0) & (df_sig['min'] == 237)]
            df_min = df_min[df_min['TICKER'].isin(valid_ts_codes.index)]
            df_min = df_min[df_min['min'] <= end_time]

            df_min[['open', 'close', 'high', 'low']] = df_min[['open', 'close', 'high', 'low']].replace(0,np.nan)
            df_min = df_min.sort_values(by=['TICKER','min']).reset_index(drop=True)
            for c in ['open', 'close', 'high', 'low']:
                df_min[c] = df_min.groupby('TICKER')[c].ffill()

            for col in ['open', 'close', 'high', 'low']:
                for i in [1, 2, 3,4]:
                    df_min[f'{col}_{i}'] = df_min.groupby('TICKER')[col].shift(i)

            use_col = ['open', 'close', 'high', 'low',
                       'open_1', 'close_1', 'high_1', 'low_1',
                       'open_2', 'close_2', 'high_2', 'low_2',
                       'open_3', 'close_3', 'high_3', 'low_3',
                       'open_4', 'close_4', 'high_4', 'low_4']
            df_min['last_5_std'] = df_min[use_col].std(axis = 1, skipna = False)
            df_min['last_5_mean'] = df_min[use_col].mean(axis = 1, skipna = False)

            # 更有波动率
            df_min['advanced_volatility'] = (df_min['last_5_std'] / df_min['last_5_mean']) ** 2

            # 收益波动比
            df_min['ret'] = df_min['close'] / df_min['close_1']
            df_min['ret_pure'] = df_min['ret']- 1
            # df_min['ret_volatility'] = df_min['ret'] / df_min['advanced_volatility']
            df_min['ret_pure_volatility'] = df_min['ret_pure'] / df_min['advanced_volatility']

            df_min['ad_vola_mean'] = df_min.groupby('TICKER')['advanced_volatility'].transform('mean')
            df_min['ad_vola_std'] = df_min.groupby('TICKER')['advanced_volatility'].transform('std')

            df_min = df_min[df_min['advanced_volatility'] > (df_min['ad_vola_mean'] + df_min['ad_vola_std'])]
            # cov_factor = \
            # df_min.groupby('ts_code').apply(lambda x:x[['advanced_volatility','ret_volatility']].cov().iloc[0,1])
            cov_pure_factor = df_min.groupby('TICKER').apply(lambda x:x[['advanced_volatility','ret_pure_volatility']].cov().iloc[0,1])
            if cov_pure_factor.empty:
                return
            result_df = cov_pure_factor.rename('cov_pure')
            # result_df = pd.concat([cov_factor.rename('cov'), cov_pure_factor.rename('cov_pure')], axis = 1)
            result_df = result_df.reset_index().rename(columns = {'index':'TICKER'})
            result_df['DATE'] = file.replace('.feather','')
            return result_df, pid

        result_list = Parallel(n_jobs=12)(delayed(main_func)(file) for file in
                                         tqdm(file_list, desc='Calculating Min Data'))
        result_list=[x for x in result_list if x is not None]
        pid_list = [x[1] for x in result_list]
        df_ls = [x[0] for x in result_list]
        kill_mutil_process(list(set(pid_list)))

        df_new = pd.concat(df_ls)

        df_old = pd.concat([df_old, df_new]).sort_values(by=['DATE', 'TICKER'])
        feather.write_dataframe(df_old, os.path.join(self.tmp_path, 'inner_day_volatility.feather'))

    def cal_daily_factors(self):
        basic_factor = feather.read_dataframe(os.path.join(self.tmp_path, 'inner_day_volatility.feather'))

        daily = feather.read_dataframe(DataPath.daily_path,columns=['TICKER', 'DATE'])
        daily = daily[(daily['DATE'] <= self.end_date) & (daily['DATE'] >= self.start_date)]
        print(daily)
        ticker_ls = daily['TICKER'].drop_duplicates()
        ticker_ls = ticker_ls[ticker_ls.apply(lambda x: x[-2:] in ['SZ', 'SH'])]
        daily = daily[daily['TICKER'].isin(ticker_ls)]

        daily = daily.sort_values(by = ['TICKER','DATE'])
        basic_factor = pd.merge(daily, basic_factor, on=['TICKER','DATE'], how='left')

        factor_mean = basic_factor.set_index('DATE').groupby('TICKER')['cov_pure'].apply(lambda x:
                                                                             x.rolling(20, min_periods = 10).mean())
        factor_std = basic_factor.set_index('DATE').groupby('TICKER')['cov_pure'].apply(lambda x:
                                                                             x.rolling(20, min_periods = 10).std())
        # factor_climb_high = (factor_mean - factor_std) / 2
        factor_climb_high = (factor_mean + factor_std) / 2

        factor_climb_high = factor_climb_high.reset_index().rename(columns={'cov_pure': '勇攀高峰_pure'})
        factor_climb_high = factor_climb_high.apply(lambda x:x.replace([np.inf, -np.inf],np.nan))
        # print(factor_climb_high)
        return factor_climb_high
        # feather.write_dataframe(factor_climb_high[['TICKER','DATE','勇攀高峰_pure']],
        #                         os.path.join(self.output_path, '勇攀高峰_pure.feather'))
        # feather.write_dataframe(factor_climb_high[['TICKER','DATE','勇攀高峰_pure']],
        #                         os.path.join(DataPath.save_path_update, '勇攀高峰_pure.feather'))

    def run(self):
        self.inner_day_volatility()
        data = self.cal_daily_factors()
        return data


def update(today='20250822'):
    if os.path.exists(os.path.join(DataPath.save_path_update, '勇攀高峰_pure.feather')):
        old = feather.read_dataframe(os.path.join(DataPath.save_path_update, '勇攀高峰_pure.feather'))
        start_date = sorted(list(old.DATE.unique()))[-55:][0]
        obj = Climb_High(start=start_date, end=today, output_path=DataPath.save_path_old)
        # print(self.start_date)
        data = obj.run()
        test = old.merge(data, on=['DATE', 'TICKER'], how='inner').dropna().tail(5)
        if np.isclose(test.iloc[:, 2], test.iloc[:, 3]).all():
            data = data[data.DATE > old.DATE.max()]
            old = pd.concat([old, data]).reset_index(drop=True)
            print(old)
            feather.write_dataframe(old, os.path.join(DataPath.save_path_update, '勇攀高峰_pure.feather'))
            feather.write_dataframe(old, os.path.join(DataPath.factor_out_path, '勇攀高峰_pure.feather'))
        else:
            print('数据检查更新出错!')
            exit()
    else:
        obj = Climb_High(start='20200101', end=today, output_path=DataPath.save_path_old)
        data = obj.run()
        feather.write_dataframe(data, os.path.join(DataPath.save_path_old, '勇攀高峰_pure.feather'))
        feather.write_dataframe(data, os.path.join(DataPath.save_path_update, '勇攀高峰_pure.feather'))

if __name__ == '__main__':
    update()

