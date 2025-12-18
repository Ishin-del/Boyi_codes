import os
import time

import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

import warnings
import gc
import sys
sys.path.append('../')


warnings.filterwarnings(action='ignore')
# 用于计算海通176日内半小时特征 逐笔分大小单类 120
class haitong_176_tic_split_deals(object):
    def __init__(self, datapath=None, savepath=None, start='20240102', end='20250801', job_use=1):
        self.datapath_sh = datapath or r'\\Desktop-79nue61\sh'
        self.datapath_sz = datapath or r'\\Desktop-79nue61\sz'

        self.savepath = savepath or r'\\DESKTOP-79NUE61\tianyixin\原始数据\haitong_176_tic_split_deals'
        os.makedirs(self.savepath,exist_ok=True)
        self.start = start
        self.end = end
        self.citic = feather.read_dataframe(r"Z:\local_data\Data_Storage\citic_code.feather")
        self.citic['DATE'] = self.citic['DATE'].astype(str)
        self.citic = self.citic[(self.citic['DATE'] <= self.end) & (self.citic['DATE'] >= self.start)]
        self.citic = self.citic.rename(columns={'citic_code': 'citic'})
        self.citic = self.citic.reset_index(drop=True)
        self.job_use = job_use

    def limit(self, processed):
        median_ = processed.median()
        MAD = np.abs(processed - median_).median()

        left = median_ - 3 * 1.483 * MAD
        right = median_ + 3 * 1.483 * MAD
        left_limit = median_ - 3.5 * 1.483 * MAD
        right_limit = median_ + 3.5 * 1.483 * MAD

        if np.isclose(left - left_limit, 0.0):
            processed[processed < left] = left
        else:
            limit = np.abs(left_limit - left)
            processed[processed < left] = left - limit * (left - processed[processed < left]) / (
                    left - processed[processed < left]).max()

        if np.isclose(right_limit - right, 0.0):
            processed[processed > right] = right
        else:
            limit = np.abs(right_limit - right)
            processed[processed > right] = right + limit * (processed[processed > right] - right) / (
                    processed[processed > right] - right).max()
        return processed

    def citic_fillna(self, x):
        cols = list(np.setdiff1d(x.columns, ['TICKER', 'DATE', 'citic', 'timestamp']))
        x[cols] = x[cols].fillna(x[cols].median())
        return x

    def zscore(self, z):
        z = (z - z.mean()) / z.std()
        return z

    def cal_and_save(self, date):
        warnings.filterwarnings(action='ignore')

        sh_path = os.path.join(self.datapath_sh, f'{date}_trade_sh.feather')
        sz_path = os.path.join(self.datapath_sz, f'{date}_trade_sz.feather')

        dates = os.path.split(sh_path)[-1][:8]
        sh = feather.read_dataframe(sh_path)
        sz = feather.read_dataframe(sz_path)

        sz['TransactTime'] = sz['TickTime']//100000
        sh['TransactTime'] = sh['TickTime']//100000
        # sz['TransactTime'] = ((sz['TransactTime'] - int(dates) * 1000000000) / 100000).astype(int)

        sz = sz[(930 <= sz['TransactTime']) & (sz['TransactTime'] <= 1456)]  # 选择全天数据
        sz = sz[['TICKER','BuyOrderNo', 'SellOrderNo', 'TransactTime','Price', 'Qty','TradeMoney','TickBSFlag']]
        sz.columns = ['TICKER', 'BuyNo', 'SellNo', 'TradeTime', 'Price', 'TradeQty', 'TradeMoney','BS_flag']
        sz = sz[['TICKER', 'TradeTime', 'BuyNo', 'SellNo', 'TradeMoney', 'TradeQty', 'Price']]
        sz = sz[sz['TICKER'].str[:2].isin(['00', '30'])]

        # 上交所
        # sh['TradeTime'] = ((sh['TradeTime'] - int(dates) * 1000000000) / 100000).astype(int)
        sh = sh[(930 <= sh['TransactTime']) & (sh['TransactTime'] <= 1456)]  # 选择全天数据
        sh = sh[['TICKER', 'BuyOrderNo', 'SellOrderNo', 'TransactTime', 'Price', 'Qty', 'TradeMoney', 'TickBSFlag']]
        sh.columns = ['TICKER', 'BuyNo', 'SellNo', 'TradeTime', 'Price', 'TradeQty', 'TradeMoney', 'BS_flag']
        sh = sh[['TICKER', 'TradeTime', 'BuyNo', 'SellNo', 'TradeMoney', 'TradeQty', 'Price']]
        sh = sh[sh['TICKER'].str[:2].isin(['60', '68'])]

        data = pd.concat([sz, sh]).reset_index(drop=True)
        del sz, sh
        gc.collect()

        data['last_Price'] = data.groupby('TICKER')['Price'].shift(1)
        data['last_Price'] = data.groupby('TICKER')['last_Price'].fillna(method='bfill')

        spt = [930, 1000, 1030, 1100, 1130, 1330, 1400, 1430, 1458]
        data['timestamp'] = 0
        for i in range(1, len(spt)):
            data.loc[(data['TradeTime'] <= spt[i]) & (data['TradeTime'] >= spt[i - 1]), 'timestamp'] = i

        # 给data打标签

        # 划分小，中123，大单（买卖两个方向） 按分位数划分
        # columns = ['TICKER', 'TradeTime', 'BuyNo', 'SellNo', 'TradeMoney', 'TradeQty', 'Price', 'BS_flag', 'last_Price','timestamp']

        def small_threshold(x):
            return np.percentile(x, 20)

        def mid1_threshold(x):
            return np.percentile(x, 40)

        def mid2_threshold(x):
            return np.percentile(x, 60)

        def mid3_threshold(x):
            return np.percentile(x, 80)

        # data_by_buy_side = data[data['BS_flag'] == 1]
        data_sum_by_buy_side = data.groupby(['TICKER', 'BuyNo']).agg({'TradeMoney': 'sum'}).reset_index(drop=False)
        data_buy_small = data_sum_by_buy_side.groupby(['TICKER']).agg({'TradeMoney': small_threshold}).reset_index(
            drop=False)
        data_buy_mid1 = data_sum_by_buy_side.groupby(['TICKER']).agg({'TradeMoney': mid1_threshold}).reset_index(
            drop=False)
        data_buy_mid2 = data_sum_by_buy_side.groupby(['TICKER']).agg({'TradeMoney': mid2_threshold}).reset_index(
            drop=False)
        data_buy_mid3 = data_sum_by_buy_side.groupby(['TICKER']).agg({'TradeMoney': mid3_threshold}).reset_index(
            drop=False)

        buy_thres = pd.merge(data_buy_small, data_buy_mid1, how='outer', on=['TICKER'], suffixes=('_small', '_mid1'))
        buy_thres = pd.merge(buy_thres, data_buy_mid2, how='outer', on=['TICKER'])
        buy_thres = pd.merge(buy_thres, data_buy_mid3, how='outer', on=['TICKER'], suffixes=('_mid2', '_mid3'))

        spt_buy = pd.merge(data_sum_by_buy_side, buy_thres)
        spt_buy['TradeMoney_small'] = spt_buy['TradeMoney'] - spt_buy['TradeMoney_small'] > 0
        spt_buy['TradeMoney_mid1'] = spt_buy['TradeMoney'] - spt_buy['TradeMoney_mid1'] > 0
        spt_buy['TradeMoney_mid2'] = spt_buy['TradeMoney'] - spt_buy['TradeMoney_mid2'] > 0
        spt_buy['TradeMoney_mid3'] = spt_buy['TradeMoney'] - spt_buy['TradeMoney_mid3'] > 0

        spt_buy['size'] = spt_buy[
            ['TradeMoney_small', 'TradeMoney_mid1', 'TradeMoney_mid2', 'TradeMoney_mid3']].sum(axis=1)
        spt_buy['size'] += 1
        spt_buy = spt_buy[['TICKER', 'BuyNo', 'size']]

        data_sum_by_Sell_side = data.groupby(['TICKER', 'SellNo']).agg({'TradeMoney': 'sum'}).reset_index(drop=False)
        data_Sell_small = data_sum_by_Sell_side.groupby(['TICKER']).agg({'TradeMoney': small_threshold}).reset_index(
            drop=False)
        data_Sell_mid1 = data_sum_by_Sell_side.groupby(['TICKER']).agg({'TradeMoney': mid1_threshold}).reset_index(
            drop=False)
        data_Sell_mid2 = data_sum_by_Sell_side.groupby(['TICKER']).agg({'TradeMoney': mid2_threshold}).reset_index(
            drop=False)
        data_Sell_mid3 = data_sum_by_Sell_side.groupby(['TICKER']).agg({'TradeMoney': mid3_threshold}).reset_index(
            drop=False)

        Sell_thres = pd.merge(data_Sell_small, data_Sell_mid1, how='outer', on=['TICKER'], suffixes=('_small', '_mid1'))
        Sell_thres = pd.merge(Sell_thres, data_Sell_mid2, how='outer', on=['TICKER'])
        Sell_thres = pd.merge(Sell_thres, data_Sell_mid3, how='outer', on=['TICKER'], suffixes=('_mid2', '_mid3'))

        spt_Sell = pd.merge(data_sum_by_Sell_side, Sell_thres)
        spt_Sell['TradeMoney_small'] = spt_Sell['TradeMoney'] - spt_Sell['TradeMoney_small'] > 0
        spt_Sell['TradeMoney_mid1'] = spt_Sell['TradeMoney'] - spt_Sell['TradeMoney_mid1'] > 0
        spt_Sell['TradeMoney_mid2'] = spt_Sell['TradeMoney'] - spt_Sell['TradeMoney_mid2'] > 0
        spt_Sell['TradeMoney_mid3'] = spt_Sell['TradeMoney'] - spt_Sell['TradeMoney_mid3'] > 0

        spt_Sell['size'] = spt_Sell[
            ['TradeMoney_small', 'TradeMoney_mid1', 'TradeMoney_mid2', 'TradeMoney_mid3']].sum(
            axis=1)
        spt_Sell['size'] += 1
        spt_Sell = spt_Sell[['TICKER', 'SellNo', 'size']]

        # 要选择的股票
        pattern = data[['TICKER', 'timestamp']].drop_duplicates(keep='first').reset_index(drop=True)
        tics = pattern.groupby('TICKER').count().reset_index(drop=False)
        tics = tics[tics['timestamp'] == 8]
        pattern = pattern[pattern['TICKER'].isin(tics['TICKER'])].reset_index(drop=True)
        pattern['DATE'] = dates

        # print('开始计算因子')

        def return_corr_amount(x):
            corrs = x[['diff_TradeMoney', 'min_ret']].corr().loc['diff_TradeMoney', 'min_ret']
            tick = x['TICKER'].iloc[0]
            tm = x['timestamp'].iloc[0]
            xx = pd.DataFrame({'TICKER': tick, 'timestamp': tm, 'corrs': corrs}, index=[0])
            return xx

        # 计算大/中3/中2/中1/小买/卖/买卖差额金额及占比等
        def calculate_mid_buy_sell_ratio(data, spt_buy, spt_Sell, pattern, dates, size_type='中2'):
            # 定义一个字典来映射 size_type 到 size
            size_mapping = {
                '大': 5,
                '中3': 4,
                '中2': 3,
                '中1': 2,
                '小': 1
            }
            # 获取 size 值
            size = size_mapping.get(size_type, -1)

            mid_buy = pd.merge(data, spt_buy[(spt_buy['size'] == size)][['TICKER', 'BuyNo']], on=['TICKER', 'BuyNo'],
                               how='inner').reset_index(drop=True)
            mid_sell = pd.merge(data, spt_Sell[spt_Sell['size'] == size][['TICKER', 'SellNo']],
                                on=['TICKER', 'SellNo'],
                                how='inner').reset_index(drop=True)
            # 收益相关性字段
            stamp_series = mid_buy[['TradeTime', 'timestamp']].drop_duplicates(keep='last').reset_index(drop=True)
            stamp_series = stamp_series.sort_values(by=['TradeTime']).reset_index(drop=True)

            # 买单收益相关性
            mid_buy_corrs = mid_buy[['TICKER', 'TradeTime', 'BuyNo', 'TradeMoney', 'TradeQty', 'Price']]
            mid_buy_ret_series = mid_buy_corrs.groupby(['TICKER', 'TradeTime'])['Price'].apply(
                lambda x: x.iloc[-1]).reset_index(drop=False)
            mid_buy_ret_series['min_ret'] = (mid_buy_ret_series['Price'] / mid_buy_ret_series.groupby('TICKER')[
                'Price'].shift(
                1) - 1) * 100
            mid_buy_ret_series['min_ret'] = mid_buy_ret_series['min_ret'].fillna(0)

            mid_buy_corrs_amount = mid_buy_corrs.groupby(['TICKER', 'TradeTime']).agg(
                {'TradeMoney': 'sum'}).reset_index(
                drop=False)
            mid_buy_corrs_all = pd.merge(mid_buy_corrs_amount, mid_buy_ret_series[['TICKER', 'TradeTime', 'min_ret']])
            mid_buy_corrs_all = pd.merge(mid_buy_corrs_all, stamp_series, on=['TradeTime']).reset_index(drop=True)
            mid_buy_corrs_all = mid_buy_corrs_all.groupby(['TICKER', 'timestamp']).apply(
                lambda x: x[['TradeMoney', 'min_ret']].corr().iloc[0, 1]).reset_index(drop=False)
            mid_buy_corrs_all.columns = ['TICKER', 'timestamp', f'{size_type}买单金额收益相关性']

            # 卖单收益相关性
            mid_sell_corrs = mid_sell[['TICKER', 'TradeTime', 'SellNo', 'TradeMoney', 'TradeQty', 'Price']]

            mid_sell_corrs_amount = mid_sell_corrs.groupby(['TICKER', 'TradeTime']).agg(
                {'TradeMoney': 'sum'}).reset_index(
                drop=False)
            mid_sell_corrs_all = pd.merge(mid_sell_corrs_amount, mid_buy_ret_series[['TICKER', 'TradeTime', 'min_ret']])
            mid_sell_corrs_all = pd.merge(mid_sell_corrs_all, stamp_series, on=['TradeTime']).reset_index(drop=True)
            mid_sell_corrs_all = mid_sell_corrs_all.groupby(['TICKER', 'timestamp']).apply(
                lambda x: x[['TradeMoney', 'min_ret']].corr().iloc[0, 1]).reset_index(drop=False)
            mid_sell_corrs_all.columns = ['TICKER', 'timestamp', f'{size_type}卖单金额收益相关性']

            # 买卖单差额收益相关性
            diff_corr_mount = pd.merge(mid_sell_corrs_amount, mid_buy_corrs_amount, on=['TICKER', 'TradeTime'],
                                       suffixes=('_sell', '_buy'), how='outer').fillna(0).reset_index(drop=True)
            diff_corr_mount['diff_TradeMoney'] = diff_corr_mount['TradeMoney_buy'] - diff_corr_mount[
                'TradeMoney_sell']
            diff_corr_mount = pd.merge(diff_corr_mount[['TICKER', 'TradeTime', 'diff_TradeMoney']],
                                       mid_buy_ret_series[['TICKER', 'TradeTime', 'min_ret']],
                                       on=['TICKER', 'TradeTime']).reset_index(drop=True)
            diff_corr_mount = pd.merge(diff_corr_mount, stamp_series, on=['TradeTime'], how='left').reset_index(
                drop=True)
            diff_corr_mount = diff_corr_mount.fillna(0)
            diff_corr_mount = diff_corr_mount[['TICKER', 'timestamp', 'diff_TradeMoney', 'min_ret']]
            diff_sell_corrs_all = diff_corr_mount.groupby(['TICKER', 'timestamp']).apply(
                return_corr_amount).reset_index(drop=True)


            diff_sell_corrs_all.columns = ['TICKER', 'timestamp', f'{size_type}买{size_type}卖单金额收益相关性']

            # 合并买卖单
            all = pd.concat([mid_buy, mid_sell])
            amount = all.groupby(['TICKER', 'timestamp']).agg({'TradeMoney': 'sum'}).reset_index(drop=False)

            # 统计买卖总金额和单均金额
            mid_buy_amount = mid_buy.groupby(['TICKER', 'timestamp']).agg(TradeMoney=('TradeMoney', 'sum'),
                                                                          count=('TradeMoney', 'count')).reset_index(
                drop=False)
            # 计算全天的总交易金额
            total_amount_for_the_day = mid_buy_amount.groupby('TICKER')['TradeMoney'].sum().reset_index(drop=False)
            total_amount_for_the_day.columns = ['TICKER', 'TradeMoney_all_day']

            mid_sell_amount = mid_sell.groupby(['TICKER', 'timestamp']).agg(TradeMoney=('TradeMoney', 'sum'),
                                                                            count=('TradeMoney', 'count')).reset_index(
                drop=False)

            # 合并计算买卖单金额差额
            mid_diff_amount = pd.merge(mid_buy_amount, mid_sell_amount, on=['TICKER', 'timestamp'],
                                       how='outer', suffixes=('_buy', '_sell')).fillna(0)
            mid_diff_amount['TradeMoney'] = mid_diff_amount['TradeMoney_buy'] - mid_diff_amount['TradeMoney_sell']
            mid_diff_amount['count'] = (mid_diff_amount['count_buy'] + mid_diff_amount['count_sell']) / 2
            mid_diff_amount = mid_diff_amount[['TICKER', 'timestamp', 'TradeMoney', 'count']]
            # 买单金额占比和买单单均金额占比
            mid_buy_amount_ratio = pd.merge(amount, mid_buy_amount, on=['TICKER', 'timestamp'], how='left',
                                            suffixes=('_all', '_mid')).reset_index(drop=True)
            mid_buy_amount_ratio[f'{size_type}买单金额占比'] = mid_buy_amount_ratio['TradeMoney_mid'] / \
                                                               mid_buy_amount_ratio[
                                                                   'TradeMoney_all']
            mid_buy_amount_ratio = pd.merge(mid_buy_amount_ratio, total_amount_for_the_day, on=['TICKER'], how='left')

            mid_buy_amount_ratio[f'{size_type}买单金额占比(占全天)'] = mid_buy_amount_ratio['TradeMoney_mid'] / \
                                                                       mid_buy_amount_ratio[
                                                                           'TradeMoney_all_day']
            mid_buy_amount_ratio[f'{size_type}买单单均金额占比'] = mid_buy_amount_ratio[f'{size_type}买单金额占比'] / \
                                                                   mid_buy_amount_ratio[
                                                                       'count']

            mid_buy_avg_amount_ratio = mid_buy_amount_ratio[['TICKER', 'timestamp', f'{size_type}买单单均金额占比']]
            mid_buy_avg_amount_ratio['DATE'] = dates
            mid_buy_avg_amount_ratio = pd.merge(pattern, mid_buy_avg_amount_ratio, how='left', on=list(pattern.columns))

            mid_buy_amount_one_day_ratio = mid_buy_amount_ratio[
                ['TICKER', 'timestamp', f'{size_type}买单金额占比(占全天)']]
            mid_buy_amount_one_day_ratio['DATE'] = dates
            mid_buy_amount_one_day_ratio = pd.merge(pattern, mid_buy_amount_one_day_ratio, how='left',
                                                    on=list(pattern.columns))

            mid_buy_amount_ratio = mid_buy_amount_ratio[['TICKER', 'timestamp', f'{size_type}买单金额占比']]
            mid_buy_amount_ratio['DATE'] = dates
            mid_buy_amount_ratio = pd.merge(pattern, mid_buy_amount_ratio, how='left', on=list(pattern.columns))

            # 卖单金额占比和卖单单均金额占比
            mid_sell_amount_ratio = pd.merge(amount, mid_sell_amount, on=['TICKER', 'timestamp'], how='left',
                                             suffixes=('_all', '_mid')).reset_index(drop=True)
            mid_sell_amount_ratio[f'{size_type}卖单金额占比'] = mid_sell_amount_ratio['TradeMoney_mid'] / \
                                                                mid_sell_amount_ratio[
                                                                    'TradeMoney_all']

            mid_sell_amount_ratio = pd.merge(mid_sell_amount_ratio, total_amount_for_the_day, on=['TICKER'], how='left')
            mid_sell_amount_ratio[f'{size_type}卖单金额占比(占全天)'] = mid_sell_amount_ratio[
                                                                            'TradeMoney_mid'] / mid_sell_amount_ratio[
                                                                            'TradeMoney_all_day']

            mid_sell_amount_ratio[f'{size_type}卖单单均金额占比'] = mid_sell_amount_ratio[f'{size_type}卖单金额占比'] / \
                                                                    mid_sell_amount_ratio[
                                                                        'count']

            mid_sell_avg_amount_ratio = mid_sell_amount_ratio[['TICKER', 'timestamp', f'{size_type}卖单单均金额占比']]
            mid_sell_avg_amount_ratio['DATE'] = dates
            mid_sell_avg_amount_ratio = pd.merge(pattern, mid_sell_avg_amount_ratio, how='left',
                                                 on=list(pattern.columns))

            mid_sell_amount_one_day_ratio = mid_sell_amount_ratio[
                ['TICKER', 'timestamp', f'{size_type}卖单金额占比(占全天)']]
            mid_sell_amount_one_day_ratio['DATE'] = dates
            mid_sell_amount_one_day_ratio = pd.merge(pattern, mid_sell_amount_one_day_ratio, how='left',
                                                     on=list(pattern.columns))

            mid_sell_amount_ratio = mid_sell_amount_ratio[['TICKER', 'timestamp', f'{size_type}卖单金额占比']]
            mid_sell_amount_ratio['DATE'] = dates
            mid_sell_amount_ratio = pd.merge(pattern, mid_sell_amount_ratio, how='left', on=list(pattern.columns))

            # 买卖差额占比
            mid_diff_amount_ratio = pd.merge(amount, mid_diff_amount, on=['TICKER', 'timestamp'], how='left',
                                             suffixes=('_all', '_mid')).reset_index(drop=True)
            mid_diff_amount_ratio[f'{size_type}买{size_type}卖差额占比'] = mid_diff_amount_ratio['TradeMoney_mid'] / \
                                                                           mid_diff_amount_ratio['TradeMoney_all']

            mid_diff_amount_ratio = pd.merge(mid_diff_amount_ratio, total_amount_for_the_day, on=['TICKER'], how='left')
            mid_diff_amount_ratio[f'{size_type}买{size_type}卖差额占比(占全天)'] = mid_diff_amount_ratio[
                                                                                       'TradeMoney_mid'] / \
                                                                                   mid_diff_amount_ratio[
                                                                                       'TradeMoney_all_day']

            mid_diff_amount_one_day_ratio = mid_diff_amount_ratio[
                ['TICKER', 'timestamp', f'{size_type}买{size_type}卖差额占比(占全天)']]
            mid_diff_amount_one_day_ratio['DATE'] = dates
            mid_diff_amount_one_day_ratio = pd.merge(pattern, mid_diff_amount_one_day_ratio, how='left',
                                                     on=list(pattern.columns))

            mid_diff_amount_ratio = mid_diff_amount_ratio[
                ['TICKER', 'timestamp', f'{size_type}买{size_type}卖差额占比']]
            mid_diff_amount_ratio['DATE'] = dates
            mid_diff_amount_ratio = pd.merge(pattern, mid_diff_amount_ratio, how='left', on=list(pattern.columns))

            # 买卖单均金额占比差值
            # 合并买单和卖单的单均金额占比数据
            mid_diff_avg_amount_ratio = pd.merge(
                mid_buy_avg_amount_ratio[['TICKER', 'timestamp', f'{size_type}买单单均金额占比']],
                mid_sell_avg_amount_ratio[['TICKER', 'timestamp', f'{size_type}卖单单均金额占比']],
                on=['TICKER', 'timestamp'],
                how='left'
            ).reset_index(drop=True)
            # 计算单均金额占比的差值
            mid_diff_avg_amount_ratio[f'{size_type}买卖单均金额占比差值'] = mid_diff_avg_amount_ratio[
                                                                                f'{size_type}买单单均金额占比'] - \
                                                                            mid_diff_avg_amount_ratio[
                                                                                f'{size_type}卖单单均金额占比']
            mid_diff_avg_amount_ratio = mid_diff_avg_amount_ratio[
                ['TICKER', 'timestamp', f'{size_type}买卖单均金额占比差值']]
            # 添加日期信息
            mid_diff_avg_amount_ratio['DATE'] = dates
            mid_diff_avg_amount_ratio = pd.merge(pattern, mid_diff_avg_amount_ratio, how='left',
                                                 on=list(pattern.columns))

            # 买单金额及单均金额
            mid_buy_amount.rename(columns={'TradeMoney': f'{size_type}买单金额'}, inplace=True)
            mid_buy_amount[f'{size_type}买单单均金额'] = mid_buy_amount[f'{size_type}买单金额'] / mid_buy_amount[
                'count']
            mid_buy_avg_amount = mid_buy_amount[['TICKER', 'timestamp', f'{size_type}买单单均金额']]
            mid_buy_avg_amount['DATE'] = dates
            mid_buy_avg_amount = pd.merge(pattern, mid_buy_avg_amount, how='left', on=list(pattern.columns))
            mid_buy_amount = mid_buy_amount[['TICKER', 'timestamp', f'{size_type}买单金额']]
            mid_buy_amount['DATE'] = dates
            mid_buy_amount = pd.merge(pattern, mid_buy_amount, how='left', on=list(pattern.columns))

            # 卖单金额
            mid_sell_amount.rename(columns={'TradeMoney': f'{size_type}卖单金额'}, inplace=True)
            mid_sell_amount[f'{size_type}卖单单均金额'] = mid_sell_amount[f'{size_type}卖单金额'] / mid_sell_amount[
                'count']
            mid_sell_avg_amount = mid_sell_amount[['TICKER', 'timestamp', f'{size_type}卖单单均金额']]
            mid_sell_avg_amount['DATE'] = dates
            mid_sell_avg_amount = pd.merge(pattern, mid_sell_avg_amount, how='left', on=list(pattern.columns))
            mid_sell_amount = mid_sell_amount[['TICKER', 'timestamp', f'{size_type}卖单金额']]
            mid_sell_amount['DATE'] = dates
            mid_sell_amount = pd.merge(pattern, mid_sell_amount, how='left', on=list(pattern.columns))

            # 买卖单差额
            mid_diff_amount.rename(columns={'TradeMoney': f'{size_type}买{size_type}卖差额'}, inplace=True)
            mid_diff_amount[f'{size_type}买{size_type}卖单均金额差值'] = mid_diff_amount[
                                                                             f'{size_type}买{size_type}卖差额'] / \
                                                                         mid_diff_amount['count']
            mid_diff_avg_amount = mid_diff_amount[['TICKER', 'timestamp', f'{size_type}买{size_type}卖差额']]
            mid_diff_avg_amount['DATE'] = dates
            mid_diff_avg_amount = pd.merge(pattern, mid_diff_avg_amount, how='left', on=list(pattern.columns))
            mid_diff_amount = mid_diff_amount[['TICKER', 'timestamp', f'{size_type}买{size_type}卖单均金额差值']]
            mid_diff_amount['DATE'] = dates
            mid_diff_amount = pd.merge(pattern, mid_diff_amount, how='left', on=list(pattern.columns))

            mid_buy_corrs_all['DATE'] = dates
            mid_sell_corrs_all['DATE'] = dates
            diff_sell_corrs_all['DATE'] = dates
            mid_buy_corrs_all = pd.merge(pattern, mid_buy_corrs_all, how='left', on=list(pattern.columns))
            mid_sell_corrs_all = pd.merge(pattern, mid_sell_corrs_all, how='left', on=list(pattern.columns))
            diff_sell_corrs_all = pd.merge(pattern, diff_sell_corrs_all, how='left', on=list(pattern.columns))

            all_this_round = pd.merge(mid_buy_amount, mid_sell_amount, on=['TICKER', 'timestamp', 'DATE'], how='left')
            all_this_round = pd.merge(all_this_round, mid_diff_amount, on=['TICKER', 'timestamp', 'DATE'], how='left')
            all_this_round = pd.merge(all_this_round, mid_buy_avg_amount, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, mid_sell_avg_amount, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, mid_diff_avg_amount, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, mid_buy_amount_ratio, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, mid_sell_amount_ratio, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, mid_diff_amount_ratio, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, mid_buy_amount_one_day_ratio, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, mid_sell_amount_one_day_ratio, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, mid_diff_amount_one_day_ratio, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, mid_buy_avg_amount_ratio, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, mid_sell_avg_amount_ratio, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, mid_diff_avg_amount_ratio, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, mid_buy_corrs_all, on=['TICKER', 'timestamp', 'DATE'], how='left')
            all_this_round = pd.merge(all_this_round, mid_sell_corrs_all, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            all_this_round = pd.merge(all_this_round, diff_sell_corrs_all, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            return all_this_round

            # return [mid_buy_amount, mid_sell_amount, mid_diff_amount,
            #         mid_buy_avg_amount, mid_sell_avg_amount, mid_diff_avg_amount,
            #         mid_buy_amount_ratio, mid_sell_amount_ratio, mid_diff_amount_ratio,
            #         mid_buy_amount_one_day_ratio, mid_sell_amount_one_day_ratio, mid_diff_amount_one_day_ratio,
            #         mid_buy_avg_amount_ratio, mid_sell_avg_amount_ratio, mid_diff_avg_amount_ratio,mid_buy_corrs_all,mid_sell_corrs_all,diff_sell_corrs_all]

        # 计算上/下行+大/中3/中2/中1/小+买/卖/买卖差额金额占比
        def calculate_up_down_mid_buy_sell_ratio(data, spt_buy, spt_Sell, pattern, dates, size_type='中2',
                                                 up_down_type='上行'):
            # 定义一个字典来映射 size_type 到 size
            size_mapping = {
                '大': 5,
                '中3': 4,
                '中2': 3,
                '中1': 2,
                '小': 1
            }
            # 获取 size 值
            size = size_mapping.get(size_type, -1)

            # 选择上行下行
            if up_down_type == '上行':
                up_down = data[data['Price'] > data['last_Price']]
            elif up_down_type == '下行':
                up_down = data[data['Price'] < data['last_Price']]
            else:
                raise ValueError("Unsupported up_down_type provided, suggest 上行 or 下行")

            up_down_mid_buy = pd.merge(up_down, spt_buy[spt_buy['size'] == size][['TICKER', 'BuyNo']],
                                       on=['TICKER', 'BuyNo'],
                                       how='inner').reset_index(drop=True)
            up_down_mid_sell = pd.merge(up_down, spt_Sell[spt_Sell['size'] == size][['TICKER', 'SellNo']],
                                        on=['TICKER', 'SellNo'],
                                        how='inner').reset_index(drop=True)

            # 合并买卖单
            up_down_all = pd.concat([up_down_mid_buy, up_down_mid_sell])
            up_down_amount = up_down_all.groupby(['TICKER', 'timestamp']).agg({'TradeMoney': 'sum'}).reset_index(
                drop=False)

            up_down_mid_buy_amount = up_down_mid_buy.groupby(['TICKER', 'timestamp']).agg(
                {'TradeMoney': 'sum'}).reset_index(
                drop=False)
            up_down_mid_sell_amount = up_down_mid_sell.groupby(['TICKER', 'timestamp']).agg(
                {'TradeMoney': 'sum'}).reset_index(
                drop=False)
            # 合并买单和卖单金额
            up_down_mid_diff_amount = pd.merge(up_down_mid_buy_amount, up_down_mid_sell_amount,
                                               on=['TICKER', 'timestamp'],
                                               how='outer', suffixes=('_buy', '_sell')).fillna(0)
            # 计算买卖单金额差额
            up_down_mid_diff_amount['TradeMoney'] = up_down_mid_diff_amount['TradeMoney_buy'] - \
                                                     up_down_mid_diff_amount[
                                                         'TradeMoney_sell']
            up_down_mid_diff_amount = up_down_mid_diff_amount[['TICKER', 'timestamp', 'TradeMoney']]

            up_down_mid_buy_amount_ratio = pd.merge(up_down_amount, up_down_mid_buy_amount, on=['TICKER', 'timestamp'],
                                                    how='left',
                                                    suffixes=('_all', '_mid')).reset_index(drop=True)
            up_down_mid_buy_amount_ratio[f'{up_down_type}{size_type}买单金额占比'] = up_down_mid_buy_amount_ratio[
                                                                                         'TradeMoney_mid'] / \
                                                                                     up_down_mid_buy_amount_ratio[
                                                                                         'TradeMoney_all']
            up_down_mid_buy_amount_ratio = up_down_mid_buy_amount_ratio[
                ['TICKER', 'timestamp', f'{up_down_type}{size_type}买单金额占比']]
            up_down_mid_buy_amount_ratio['DATE'] = dates
            up_down_mid_buy_amount_ratio = pd.merge(pattern, up_down_mid_buy_amount_ratio, how='left',
                                                    on=list(pattern.columns))

            up_down_mid_sell_amount_ratio = pd.merge(up_down_amount, up_down_mid_sell_amount,
                                                     on=['TICKER', 'timestamp'],
                                                     how='left',
                                                     suffixes=('_all', '_mid')).reset_index(drop=True)
            up_down_mid_sell_amount_ratio[f'{up_down_type}{size_type}卖单金额占比'] = up_down_mid_sell_amount_ratio[
                                                                                          'TradeMoney_mid'] / \
                                                                                      up_down_mid_sell_amount_ratio[
                                                                                          'TradeMoney_all']
            up_down_mid_sell_amount_ratio = up_down_mid_sell_amount_ratio[
                ['TICKER', 'timestamp', f'{up_down_type}{size_type}卖单金额占比']]
            up_down_mid_sell_amount_ratio['DATE'] = dates
            up_down_mid_sell_amount_ratio = pd.merge(pattern, up_down_mid_sell_amount_ratio, how='left',
                                                     on=list(pattern.columns))

            up_down_mid_diff_amount_ratio = pd.merge(up_down_amount, up_down_mid_diff_amount,
                                                     on=['TICKER', 'timestamp'],
                                                     how='left',
                                                     suffixes=('_all', '_mid')).reset_index(drop=True)
            up_down_mid_diff_amount_ratio[f'{up_down_type}{size_type}买{size_type}卖差额占比'] = \
                up_down_mid_diff_amount_ratio[
                    'TradeMoney_mid'] / \
                up_down_mid_diff_amount_ratio[
                    'TradeMoney_all']
            up_down_mid_diff_amount_ratio = up_down_mid_diff_amount_ratio[
                ['TICKER', 'timestamp', f'{up_down_type}{size_type}买{size_type}卖差额占比']]
            up_down_mid_diff_amount_ratio['DATE'] = dates
            up_down_mid_diff_amount_ratio = pd.merge(pattern, up_down_mid_diff_amount_ratio, how='left',
                                                     on=list(pattern.columns))
            all_this_round = pd.merge(up_down_mid_buy_amount_ratio, up_down_mid_sell_amount_ratio,
                                      on=['TICKER', 'timestamp', 'DATE'], how='left')
            all_this_round = pd.merge(all_this_round, up_down_mid_diff_amount_ratio, on=['TICKER', 'timestamp', 'DATE'],
                                      how='left')
            return all_this_round

        factor_list = []
        for i in ['小', '大', '中3', '中2', '中1']:
            factor_list.append(calculate_mid_buy_sell_ratio(data, spt_buy, spt_Sell, pattern, dates, size_type=i))
            for j in ['上行', '下行']:
                factor_list.append(calculate_up_down_mid_buy_sell_ratio(data, spt_buy, spt_Sell, pattern, dates,
                                                                        size_type=i,
                                                                        up_down_type=j))
        all_this_day = pattern
        for i in range(len(factor_list)):
            all_this_day = pd.merge(all_this_day, factor_list[i], how='left', on=['TICKER', 'timestamp', 'DATE'])

        this_citic = self.citic[self.citic['DATE'] == dates].reset_index(drop=True)
        all_this_day = pd.merge(all_this_day, this_citic, on=['TICKER', 'DATE'], how='left')

        factor_columns = np.setdiff1d(all_this_day.columns, ['TICKER', 'DATE', 'timestamp', 'citic'])
        factor_columns = list(factor_columns)
        all_this_day[factor_columns] = all_this_day[factor_columns].replace([np.inf, -np.inf], np.nan)
        all_this_day = all_this_day[['TICKER', 'DATE', 'timestamp', 'citic'] + factor_columns]
        gg = all_this_day.groupby('timestamp')
        res = []
        for i in gg:
            now = i[1]
            now[factor_columns] = now[factor_columns].apply(lambda t: self.limit(t))
            res.append(now)
        res = pd.concat(res).reset_index(drop=True)
        res = res.dropna(subset=['TICKER', 'DATE', 'timestamp', 'citic']).reset_index(drop=True)

        all_this_day = res.groupby(['citic', 'timestamp']).apply(self.citic_fillna).reset_index(drop=True)  # 行业中位数填充
        all_this_day.drop('citic', axis=1, inplace=True)
        all_this_day = all_this_day.reset_index(drop=True)
        gg = all_this_day.groupby('timestamp')
        res = []
        for i in gg:
            now = i[1]
            now[factor_columns] = now[factor_columns].apply(lambda t: self.zscore(t))
            now[factor_columns] = now[factor_columns].fillna(now[factor_columns].median())
            res.append(now)
        all_this_day = pd.concat(res).reset_index(drop=True)
        all_this_day[factor_columns] = all_this_day[factor_columns].fillna(all_this_day[factor_columns].median())
        all_this_day[factor_columns] = all_this_day[factor_columns].fillna(0)
        feather.write_dataframe(all_this_day, os.path.join(self.savepath, date + '.feather'))

    def cal(self):
        os.makedirs(self.savepath, exist_ok=True)
        datels1 = os.listdir(self.datapath_sh)
        datels2 = os.listdir(self.datapath_sz)
        datels1 = [i[:8] for i in datels1 if 'trade' in i]
        datels2 = [i[:8] for i in datels2 if 'trade' in i]
        datels = [i[:8] for i in list(np.intersect1d(datels1,datels2))]
        datels = list(set(datels))

        already_date = os.listdir(self.savepath)
        already_date = [i[:8] for i in already_date]
        need_date = np.setdiff1d(datels, already_date)
        need = []
        for i in need_date:
            if i <= self.end and i >= self.start:
                need.append(i)
        if self.job_use > 1:
            Parallel(n_jobs=self.job_use)(delayed(self.cal_and_save)
                                          (date) for date in tqdm(need))
        else:
            for date in tqdm(need):
                self.cal_and_save(date)


if __name__ == '__main__':
    jbo = haitong_176_tic_split_deals(job_use=2)
    jbo.cal()
