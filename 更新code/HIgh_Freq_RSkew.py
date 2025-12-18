# -*- coding: utf-8 -*-
# Created on  2021/10/12 11:00
# @Author: Yangyu Che
# @File: Ret20_Improved.py
# @Contact: cheyangyu@126.com
# @Software: PyCharm

import os
import time
import warnings

import feather
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm

from tool_tyx.path_data import DataPath


class RSkew:
    def __init__(self, start = None, end = None,savepath=None):
        self.start_date = start #or '20250102'
        self.end_date = end #or '20250127'
        self.data_path = DataPath.to_path
        self.out_path=savepath
        # self.min_data_path = '\\\\10.36.35.85\\Data_storage\\Min_Data/'
        self.sh_min = DataPath.sh_min
        self.sz_min = DataPath.sz_min

        exist_file_list = os.listdir(self.sh_min)
        self.max_min_data_date = max([x[:10] for x in exist_file_list])
        self.look_back_period = 20


    def Generate_Calendar(self):
        warnings.filterwarnings('ignore')
        df_calendar = pd.read_csv(self.data_path+'\\calendar.csv', dtype = {'trade_date': str})
        df_calendar.rename(columns={'trade_date':'DATE'},inplace=True)
        df_calendar = df_calendar[(df_calendar['DATE'] >= self.start_date) & (df_calendar['DATE'] <= self.end_date)]
        df_calendar['month'] = df_calendar['DATE'].map(lambda x: x[:6])
        df_calendar = df_calendar.sort_values(by = ['month', 'DATE'], ascending = True)

        df_month_end = df_calendar.drop_duplicates(subset = ['month'], keep = 'last')['DATE']

        self.date_list = df_calendar['DATE']
        self.month_end_list = df_month_end

    def load_min_data(self, file_name):
        try:
            # df_min = feather.read_dataframe(self.min_data_path + file_name,columns = ['ts_code', 'trade_time', 'close', 'open']).set_index('ts_code')
            df_min_sh = feather.read_dataframe(self.sh_min + '/'+file_name,columns = ['TICKER', 'min', 'close', 'open','DATE']) #.set_index('TICKER')
            df_min_sz = feather.read_dataframe(self.sz_min + '/'+ file_name,columns = ['TICKER', 'min', 'close', 'open','DATE']) #.set_index('TICKER')
            df_min=pd.concat([df_min_sh,df_min_sz]).reset_index(drop=True)
        except:
            if file_name[:10] <= self.max_min_data_date:
                raise FileNotFoundError
            else:
                return None

        # date = df_min['trade_time'].iloc[0][:10]
        date = df_min['DATE'].iloc[0]
        # start_time = date + ' 09:30:00'
        # df_min = df_min[df_min['trade_time']!=start_time]
        df_min = df_min[df_min['min']!=930]
        df_min['return_1'] = df_min['close'] / df_min['open'] - 1
        df_min['return_2'] = df_min['return_1'] * df_min['return_1']
        df_min['return_3'] = df_min['return_2'] * df_min['return_1']

        numerator = df_min.groupby('TICKER')['return_3'].sum() * np.sqrt(
            df_min.groupby('TICKER')['return_3'].count())
        denomiator = np.power(df_min.groupby('TICKER')['return_2'].sum(), 1.5)

        Rskew_daily = numerator / denomiator

        Rskew_daily = Rskew_daily.reset_index(name = 'RSkew_daily').rename(columns = {'ts_code': 'TICKER'})
        Rskew_daily['DATE'] = date.replace('-', '')

        return Rskew_daily

    def cal_RSkew_daily(self, update = False):
        warnings.filterwarnings('ignore')
        try:
            RSkew_daily_old = feather.read_dataframe(self.out_path +'\\RSkew_daily.feather')

            # 删除一些历史数据重新计算，以保证代码的正确性
            cut_date = RSkew_daily_old['DATE'].drop_duplicates().sort_values().iloc[-20]
            RSkew_daily_old = RSkew_daily_old[RSkew_daily_old['DATE'] <= cut_date]

            exist_date = RSkew_daily_old['DATE'].unique()
        except:
            RSkew_daily_old = pd.DataFrame()
            exist_date = []

        # file_list = [x[:4] + '-' + x[4:6] + '-' + x[6:8] + '.feather' for x in np.setdiff1d(self.date_list, exist_date)]
        file_list = [x + '.feather' for x in np.setdiff1d(self.date_list, exist_date)]

        RSkew_daily_array = np.array(())
        for file_name in tqdm(file_list, desc = 'Calculating Daily Ret'):
            RSkew_daily = self.load_min_data(file_name)

            if RSkew_daily is None:
                continue
            else:
                RSkew_daily_array = np.array(RSkew_daily.columns) if RSkew_daily_array.shape[0]==0 else RSkew_daily_array
                RSkew_daily_array = np.vstack((RSkew_daily_array, RSkew_daily.values))
        RSkew_daily_df = pd.DataFrame(RSkew_daily_array[1:], columns = RSkew_daily_array[0]) if RSkew_daily_array.shape[
                                                                                                    0] > 0 else pd.DataFrame()

        RSkew_daily_old = pd.concat([RSkew_daily_old, RSkew_daily_df])
        RSkew_daily_old = RSkew_daily_old.sort_values(by = ['DATE', 'TICKER']).reset_index(drop = True)
        print(RSkew_daily_old)
        feather.write_dataframe(RSkew_daily_old, DataPath.save_path_update + '\\RSkew_daily.feather')
        feather.write_dataframe(RSkew_daily_old, self.out_path + '\\RSkew_daily.feather')
        self.RSkew_daily = RSkew_daily_old

    def cal_RSkew(self):
        warnings.filterwarnings('ignore')
        # self.RSkew_daily=feather.read_dataframe(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\2021.6-2025.8\RSkew_daily.feather')
        tqdm.pandas(desc = 'Calculating RSkew')
        self.RSkew_daily.set_index(['DATE'], inplace = True)
        RSkew_daily = self.RSkew_daily.groupby('TICKER').progress_apply(
            lambda x: x['RSkew_daily'].rolling(self.look_back_period, min_periods = 5).sum())

        RSkew_daily = RSkew_daily.reset_index(level = ['TICKER', 'DATE'])
        print(RSkew_daily)
        RSkew_daily.rename(columns={'RSkew_daily':'RSkew'},inplace=True)
        feather.write_dataframe(RSkew_daily, DataPath.save_path_update + '\\RSkew.feather')
        feather.write_dataframe(RSkew_daily, self.out_path + '\\RSkew.feather')

    def run(self):
        self.Generate_Calendar()
        self.cal_RSkew_daily()
        self.cal_RSkew()

def update(today='20250820'):
    RSkew_Improved_terminal_1 = RSkew(start='20250301', end=today, savepath=DataPath.factor_out_path)
    self = RSkew_Improved_terminal_1
    RSkew_Improved_terminal_1.run()

if __name__=='__main__':
    update()
