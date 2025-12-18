# -*- coding: utf-8 -*-
# Created on  2022/10/13 8:57
# @Author: zkh
# @File: technical_min.py
# @Contact: wangpaigum@163.com
# @Software: PyCharm


# ['Amihudilliq', 'corr_VP', "corr_VRlag", "ret_H8",
# "real_skewlarge", "corr_VPlarge", "corr_VRlaglarge"]
# 广发证券-多因子Alpha系列报告之（四十一）：高频价量数据的因子化方法

import os
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib.parallel import Parallel, delayed, cpu_count
import warnings

# 用于计算用到分钟频率的因子(附带有带半衰期的模板，带半衰期需要在创建类的时候送入half参数)
# 根据不同的算法调板子即可   要修改的部分
# 1.类名
# 2.init中的factor_name
# 3.__daily_calcuation()方法中get_date前面要计算的股票日基础因子列表
# 4.cal()方法中Fct_list列表中填要计算的因子名
# 5.填写两个计算的过程函数：__daily_calcuation()方法中get_date和cal()方法中get_factor函数

class technical_min(object):
    def __init__(self, start='20220104', end='20251119', path=None, savepath=None, data_path=None,
                 split_sec=None):

        self.start = start
        self.end = end
        self.path = path or '\\\\192.168.1.210\\Data_Storage2\\'
        self.savepath = savepath or 'E:\\vp_zkh\\vp_factors\\technical_min\\'
        self.data_path = data_path or 'D:\\zkh\\Price_and_vols_at_auction_time\\concat_daily\\'
        self.daily_list_file = os.listdir(self.data_path)
        self.split_sec = split_sec or 3
        calendar = pd.read_csv(self.path + 'calendar.csv', dtype={'trade_date': str})
        self.daily_list = calendar[(calendar['trade_date'] >= self.start) & (calendar['trade_date'] <= self.end)]['trade_date']

    def __daily_calcuation(self):
        if os.path.exists(self.savepath + 'basic\\basic_data.feather'):
            df_old = feather.read_dataframe(self.savepath + 'basic\\basic_data.feather')
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []

        max_min_date = max(self.daily_list_file)[:8]
        date_list = np.setdiff1d(self.daily_list, exist_date)

        fact_lis = ['Amihudilliq_D', 'corr_VP_D', "corr_VRlag_D", "ret_H8_D",
                    "real_skewlarge_D", "corr_VPlarge_D", "corr_VRlaglarge_D"]  # 要计算的基础日因子的列表，可能有多个
        self.fact_lis = fact_lis
        data_path = self.data_path

        def get_date(x):
            warnings.filterwarnings(action='ignore')
            # 日频的基础数据调用这个函数获得
            amid = x[x['volume_diff'] > 0]
            a = amid['volume_diff'].mean()/max(amid['volume_diff'].std(),0.001)
            b = x['close'].corr(x['volume_diff'])
            c = x['volume_diff'].corr(x['retlag'])
            lh = x[x['time'] > x['time'].max()//2]
            if not lh.empty:
                lh = lh.sort_values(by='time')
                gg = lh['close'].tolist()[-1]
                dd = lh['open'].tolist()[0]
                d = gg / dd - 1
            else:
                d = np.nan
            bigvol = x.sort_values(by="volume_diff")
            bigvol = bigvol.reset_index(drop=True)
            bigvol = bigvol.iloc[0:bigvol.shape[0] // 3, :]
            e = bigvol['volume_diff'].skew()
            g = bigvol['volume_diff'].mean() / max(bigvol['volume_diff'].std(),0.001)
            f = g/a
            # b = '某只股票某天要计算的最终结果2' 计算结果就按照abc一次排列到返回的列表中date的后面(ls)
            # 计算的结果相当于fact_lis
            ls = [a, b, c, d, e, f, g]
            return [x['TICKER'].unique()[0],
                    np.array(x['DATE'].values[0])] + ls

        def get_daily(date):
            # for循环tqdm
            warnings.filterwarnings(action='ignore')
            filename = date + '.feather'
            if not os.path.exists(data_path + filename):
                if date <= max_min_date:
                    raise FileNotFoundError
            else:
                to_day = np.array(['TICKER', 'DATE'] + fact_lis)
                dt = feather.read_dataframe(
                    data_path + filename)  # 所有的数据读进来截取到92459950
                dt = dt[dt['tradetime'] < 92459950]
                dt = dt[dt['tradetime'] >= 92000000]
                dt['sec'] = dt['tradetime']//1000
                seclist = dt['sec'].unique().tolist()
                seclist.sort()
                stampmap = {}
                for x in range(0,len(seclist),self.split_sec):
                    for j in range(self.split_sec):
                        stampmap[seclist[min(x+j,len(seclist) - 1)]] = x//self.split_sec

                dt['time_range'] = dt['sec'].map(stampmap)
                dt = dt.sort_values(by=['TICKER','ApplSeqNum']).reset_index(drop=True)

                split_by_sec = dt.groupby(['TICKER','time_range']).agg(
                    {'当前撮合成交量': ['first', 'last'], '当前撮合价格': ['first', 'last', 'max', 'min']}).reset_index(
                    drop=False)
                split_by_sec.columns = ['TICKER','time', 'vol_start', 'vol_end', 'open', 'close', 'high', 'low']
                split_by_sec['volume_diff'] = split_by_sec['vol_end'] - split_by_sec['vol_start']
                split_by_sec['volatility'] = 100 * (split_by_sec['high'] / split_by_sec['low'] - 1)

                split_by_sec['ret'] = split_by_sec['close'] / split_by_sec['open'] - 1
                split_by_sec['ret'] = split_by_sec['ret'].replace([np.inf, -np.inf], np.nan)
                split_by_sec = split_by_sec.sort_values(by=['TICKER', 'time'])
                split_by_sec['retlag'] = split_by_sec.groupby('TICKER')['ret'].shift(1)
                split_by_sec = split_by_sec.dropna()
                split_by_sec['DATE'] = date
                dt = split_by_sec.groupby('TICKER').apply(get_date)
                dd = []
                for i in dt:
                    if i:
                        dd.append(i)
                to_day = np.vstack((to_day, dd))
                del dt, dd
                to_day = pd.DataFrame(to_day[1:], columns=to_day[0]) if to_day.shape != (
                    2 + len(fact_lis),) else pd.DataFrame()
                if not to_day.empty:
                    to_day['DATE'] = [''.join(i.split('-')) for i in to_day['DATE'].tolist()]
                    for i in fact_lis:
                        to_day[i] = to_day[i].astype(float)
                return to_day
            # *****************
        newdata = []
        for i in tqdm(date_list):
            newdata.append(get_daily(i))# for循环 tqdm
        if newdata:
            to_day = pd.concat(newdata).reset_index(drop=True)
            if not df_old.empty:
                for i in fact_lis:
                    df_old[i] = df_old[i].astype(float)
            df_old = pd.concat([df_old, to_day]).sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
            feather.write_dataframe(df_old, self.savepath + 'basic\\basic_data.feather')

    def cal(self):
        data = feather.read_dataframe(self.savepath + 'basic\\basic_data.feather')
        factors = np.setdiff1d(data.columns,['TICKER','DATE'])
        for c in factors:
            feather.read_dataframe(data[['TICKER','DATE',c]],self.savepath + f'factor\\{c}.feather')

    def run(self):
        """
        方便外部调用
        在统一的全部因子更新脚本中，对于每个import的类都会只运行run这个方法，以实现因子更新的目的
        """
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        if not os.path.exists(self.savepath + '程序\\'):
            os.makedirs(self.savepath + '程序\\')
        if not os.path.exists(self.savepath + 'factor\\'):
            os.makedirs(self.savepath + 'factor\\')
        if not os.path.exists(self.savepath + 'basic\\'):
            os.makedirs(self.savepath + 'basic\\')
        self.__daily_calcuation()
        self.cal()


if __name__ == '__main__':
    object = technical_min()
    # break_point = 1
    object.run()
