# -*- coding: utf-8 -*-
# Created on  2021/10/27 16:35
# @Author: Yangyu Che
# @File: High_price_trade_factor.py
# @Contact: cheyangyu@126.com
# @Software: PyCharm


import os
import time
import feather
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
from tool_tyx.data_path import DataPath


class high_price_trade:

    def __init__(self, start=None, end=None,savepath=None):
        self.data_path = DataPath.to_path
        self.output_directory = savepath

        # self.min_path = '\\\\10.36.35.85\\Data_storage\\Min_Data/'
        self.sh_min_path = DataPath.sh_min
        self.sz_min_path = DataPath.sz_min
        self.start_date = start #or '20250102'
        self.end_date = end #or '20250127'

    def Generate_Calendar(self):
        warnings.filterwarnings('ignore')
        df_calendar = pd.read_csv(self.data_path + '\calendar.csv', dtype={'trade_date': str})
        df_calendar.rename(columns={'trade_date':'DATE'},inplace=True)
        df_calendar = df_calendar[(df_calendar['DATE'] >= self.start_date) & (df_calendar['DATE'] <= self.end_date)]

        self.date_list = df_calendar['DATE']

    def cal_factors(self):

        try:
            df_old = feather.read_dataframe(
                self.output_directory + 'high_price_trade_factor.feather')

            # 删除一些历史数据重新计算，以保证代码的正确性
            cut_date = df_old['DATE'].drop_duplicates().sort_values().iloc[-20]
            df_old = df_old[df_old['DATE'] <= cut_date]
            exist_date = df_old['DATE'].unique()
        except:
            df_old = pd.DataFrame()
            exist_date = []
        file_list = os.listdir(self.sh_min_path)
        # todo:
        max_min_date = max(file_list)[:10].replace('-', '')

        date_list = np.setdiff1d(self.date_list, exist_date)
        input_columns = ['TICKER', 'min', 'close', 'volume', 'amount','DATE']
        output_columns = ['TICKER', 'min', 'close', 'volume', 'amount','DATE']

        factor_array = np.array(())
        for date in tqdm(date_list, desc='Calculating high price trade Factors'):

            # file = date[:4] + '-' + date[4:6] + '-' + date[6:8] + '.feather'
            # begin_time = file[:10] + ' 09:30:00'
            file=date+'.feather'
            try:
                df_min1 = feather.read_dataframe(self.sh_min_path +'/'+ file, columns=input_columns)
                df_min2 = feather.read_dataframe(self.sz_min_path +'/'+ file, columns=input_columns)
                df_min=pd.concat([df_min1,df_min2]).reset_index(drop=True)
            except:
                if date <= max_min_date:
                    raise FileNotFoundError
                else:
                    continue
            df_min.columns = output_columns

            df_min = df_min[df_min['volume'] > 0]
            df_min = df_min[df_min['min'] != 930]

            # daily_volume = df_min.groupby('TICKER')['vol'].sum()
            # df_min = pd.merge(df_min, daily_volume.reset_index(name = 'VOLUME'), on = ['TICKER'], how = 'right')
            #
            # daily_count = df_min.groupby('TICKER')['vol'].count()
            # df_min = pd.merge(df_min, daily_count.reset_index(name = 'T'), on = ['TICKER'], how = 'left')
            #
            #
            # weighted_close_rate = df_min.groupby('TICKER').apply(lambda x:
            #                     (np.sum(x['vol'] * x['close']) / x['VOLUME'].iloc[0]) / np.sum(x['close']) * x['T'].iloc[0])
            #
            # close_std = df_min.groupby('TICKER').apply(lambda x: np.std(x['close']) ** 3
            #                                                 if not np.isclose(np.std(x['close']), 0) else np.nan)
            #
            # skew_fenzi = df_min.groupby('TICKER').apply(lambda x:
            #                      np.sum(x['vol'] / x['VOLUME'].iloc[0] * (x['close'] - np.mean(x['close'])) ** 3))
            #
            # weighted_skewness = skew_fenzi / close_std
            #
            # volume_close_entropy = df_min.groupby('TICKER').apply(lambda x:
            #                      x['close'] * x['vol'] / (x['close'].iloc[-1] * x['VOLUME']))
            # volume_close_entropy = volume_close_entropy.reset_index(level = 'TICKER', name = 'p')
            # volume_close_entropy = volume_close_entropy.groupby('TICKER').apply(lambda x:
            #                     -1 * np.sum(x['p'] * np.log(x['p'])))
            #
            # amount_entropy = df_min.groupby('TICKER').apply(lambda x:
            #                     -1 * np.sum(x['amount'] / x['amount'].sum() * np.log(x['amount'] / x['amount'].sum())))
            #
            # df_factor = pd.concat([weighted_close_rate.rename('weighted_close_rate'),
            #                        weighted_skewness.rename('weighted_skewness'),
            #                        volume_close_entropy.rename('volume_close_entropy'),
            #                        amount_entropy.rename('amount_entropy')], axis = 1)

            df_factor = df_min.groupby('TICKER').apply(self.cal_inner_factor)

            df_factor['DATE'] = date
            df_factor.reset_index(inplace=True)

            if factor_array.shape[0] == 0:
                factor_array = np.array(df_factor.columns)
                factor_array = np.vstack((factor_array, df_factor.values))
            else:
                factor_array = np.vstack((factor_array, df_factor.values))

        # def func_1(date):
        #
        #     file = date[:4] + '-' + date[4:6] + '-' + date[6:8] + '.feather'
        #     begin_time = file[:10] + ' 09:30:00'
        #
        #     try:
        #         df_min = feather.read_dataframe(self.min_path + file, columns = input_columns)
        #     except:
        #         if date <= max_min_date:
        #             raise FileNotFoundError
        #         else:
        #             return pd.DataFrame()
        #     df_min.columns = output_columns
        #
        #     df_min = df_min[df_min['vol'] > 0]
        #     df_min = df_min[df_min['trade_time']!=begin_time]
        #
        #
        #     df_factor = df_min.groupby('TICKER').apply(self.cal_inner_factor)
        #     # g = df_min.groupby('TICKER')
        #
        #     df_factor['DATE'] = date
        #     df_factor.reset_index(inplace = True)
        #
        #     return df_factor
        #
        # new_data = Parallel(n_jobs = 8)(
        #     delayed(func_1)(date) for date in tqdm(date_list, desc = 'Calculating high price trade Factors'))

        df_factor = pd.DataFrame(factor_array[1:], columns=factor_array[0]) \
                        if factor_array.shape[0] > 0 else pd.DataFrame()
        df_old = pd.concat([df_old, df_factor]).sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
        feather.write_dataframe(df_old,
                                self.output_directory + 'high_price_trade_factor.feather')

    def cal_inner_factor(self, input_df):

        daily_volume = input_df['volume'].sum()
        close_sigma3 = input_df['close'].std() ** 3
        if np.isclose(daily_volume, 0) or np.isclose(close_sigma3, 0):
            output_series = pd.Series({'volume_close_entropy': np.nan,
                                       'weighted_close_rate': np.nan,
                                       'weighted_skewness': np.nan,
                                       'amount_entropy': np.nan
                                       })
        else:
            vol_weight = input_df['volume'] / daily_volume
            close_percent = input_df['close'] / input_df['close'].sum()
            vol_close = vol_weight * close_percent

            amt_rate = input_df['amount'] / input_df['amount'].sum()

            weighted_close_rate = np.sum(vol_close) * 240
            weighted_skewness = vol_weight * (input_df['close'] - input_df['close'].mean()) ** 3
            weighted_skewness = weighted_skewness.sum() / close_sigma3

            # vol_close_1 = vol_close / vol_close.sum()
            volume_close_entropy = -1 * np.sum(vol_close * np.log(vol_close))
            amt_entropy = -1 * np.sum(amt_rate * np.log(amt_rate))

            output_series = pd.Series({'volume_close_entropy': volume_close_entropy,
                                       'weighted_close_rate': weighted_close_rate,
                                       'weighted_skewness': weighted_skewness,
                                       'amount_entropy': amt_entropy
                                       })

        return output_series

    def run(self):
        self.Generate_Calendar()
        self.cal_factors()
        # self.cal_inner_factor()


if __name__ == '__main__':
    # today = dt.datetime.today().strftime("%Y%m%d")
    high_price_trade_terminal_1 = high_price_trade(start='20200101',end='20221231')
    self = high_price_trade_terminal_1

    start = time.time()
    high_price_trade_terminal_1.run()

    end = time.time()
    print('共计用时%f秒' % (end - start))
    #-------------------------
    print('更新：')
    high_price_trade_terminal_1 = high_price_trade(start='20221231', end='20250801')
    self = high_price_trade_terminal_1

    start = time.time()
    high_price_trade_terminal_1.run()

    end = time.time()
    print('共计用时%f秒' % (end - start))
    break_point = 1
