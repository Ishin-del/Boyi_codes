# -*- coding: utf-8 -*-
# Created on  2024/6/11 11:22
# @Author: Yangyu Che
# @File: 一视同仁.py
# @Contact: cheyangyu@126.com
# @Software: PyCharm

"""
《成交量激增与骤降时刻的对称性与“一视同仁”因子构建——多因子选股系列研究之十八》
"""

import os
import feather
import warnings
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
from joblib import Parallel, delayed
import socket
from scipy import stats
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import kill_mutil_process
"""
这个因子可以考虑两个版本
1. 完全使用正成交量时刻
2. 使用正成交量时刻判断激增和骤降后，用全体时刻凑成“五分钟”
"""

"""
估计是因为在更数据 才会跑得这么慢吧？先不动它了 让它继续跑着吧？
"""


def zscore(x):
    return (x - x.mean()) / x.std()

def fairness_weight(df):
    df = df[df['volume'] > 0]
    # todo:
    if df.shape[0] <= 50:
        return None
    df['return_std'] = df['return'].iloc[::-1].rolling(5).std().iloc[::-1]
    boxcox_vol = pd.Series(stats.boxcox(df['volume'], lmbda=None)[0], index=df.index, name='vol_boxcox')
    boxcox_vol_diff = boxcox_vol - boxcox_vol.shift(1)
    # todo：
    boxcox_mean_std = boxcox_vol_diff.mean() + boxcox_vol_diff.std()

    bright_std = df[boxcox_vol_diff > boxcox_mean_std]['return_std'].mean()
    dark_std = df[boxcox_vol_diff < boxcox_mean_std]['return_std'].mean()

    bright_rtn = df[boxcox_vol_diff > boxcox_mean_std]['return'].mean()
    dark_rtn = df[boxcox_vol_diff < boxcox_mean_std]['return'].mean()

    std_fairness = bright_std - dark_std
    rtn_fairness = bright_rtn - dark_rtn
    return pd.Series([std_fairness, rtn_fairness], index=['std_fairness', 'rtn_fairness'])


class No_Exception:

    def __init__(self, start=None, end=None,out_path=None):
        # start一般不需要输入
        self.start_date = start
        self.end_date = end
        self.sh_min=DataPath.sh_min
        self.sz_min=DataPath.sz_min
        self.data_path = DataPath.to_path #alendar.csv路径

        self.min_data_path = os.path.join(self.sh_min, 'Min_Data')
        # self.output_path = os.path.join(DataPath.out_path, r'No_Exception')
        self.output_path = out_path
        self.tmp_path = os.path.join(DataPath.tmp_path, r'一视同仁daily_result')
        os.makedirs(self.tmp_path,exist_ok=True)
        self.calendar = pd.read_csv(os.path.join(self.data_path, 'calendar.csv'))['trade_date'].astype(str)
        self.daily = pd.DataFrame()

    def minute_factor(self, date):
        """
        每日分钟数据计算因子的函数
        单独定义出来 方便多进程调用
        """
        warnings.filterwarnings(action='ignore')
        pid = os.getpid()
        real_date = date
        if os.path.exists(os.path.join(self.tmp_path, f'{date}.feather')):
            result_df = feather.read_dataframe(os.path.join(self.tmp_path, f'{date}.feather'))
        else:
            while True:
                # file = date[:4] + '-' + date[4:6] + '-' + date[6:8] + '.feather'
                file=date+'.feather'
                df_min_sz,df_min_sh=pd.DataFrame(),pd.DataFrame()
                try:
                    df_min_sz = feather.read_dataframe(os.path.join(self.sz_min, file),columns=['TICKER', 'min',
                                                                        'close', 'volume', 'open', 'high','low'])
                except FileNotFoundError:
                    pass
                try:
                    df_min_sh=feather.read_dataframe(os.path.join(self.sh_min,file),columns=['TICKER', 'min',
                                                                        'close', 'volume', 'open', 'high','low'])
                except FileNotFoundError:
                    pass
                df_min=pd.concat([df_min_sz,df_min_sh]).reset_index(drop=True)
                if df_min.empty:
                    continue
                # date = df_min['trade_time'].iloc[0][:10].replace('-', '')
                df_min = df_min[df_min['min']>930]
                # start_time = df_min['min'].min()
                # start_time = start_time[:11] + '09:31:00'
                # end_time = start_time[:11] + '14:57:00'
                start_time=1457
                end_time=1459
                df_min=df_min[(df_min['min']<start_time)]
                df_sig = df_min.groupby('TICKER').agg({'close': 'mean', 'volume': 'sum', 'min': len})
                # todo:
                valid_ts_codes = df_sig[(df_sig['close'] > 0) & (df_sig['volume'] > 0) & (df_sig['min'] == 237)]
                # valid_ts_codes = df_sig[(df_sig['volume'] > 0)]
                df_min = df_min[df_min['TICKER'].isin(valid_ts_codes.index)]
                df_min['return'] = df_min.groupby('TICKER')['close'].pct_change() + 1
                df_min['return'] = df_min['return'].replace([np.inf,-np.inf],np.nan)
                # df_min = df_min[df_min['min'] <= end_time]
                # df_min = df_min[df_min['trade_time'] >= start_time]

                # tqdm.pandas()
                inday_return = df_min.groupby('TICKER').apply(lambda x: x['close'].iloc[-1] / x['open'].iloc[0])
                inday_return = inday_return.replace([np.inf, -np.inf], np.nan)
                inday_return = inday_return.replace(0, 1)
                # tqdm.pandas()
                fairness_weight_value = df_min.groupby('TICKER').apply(fairness_weight)
                if fairness_weight_value.empty:
                    print(f'{real_date}的分钟数据异常，用前一天数据填充')
                    # 今天的数据有问题，用前一天的数据填充
                    date = self.calendar[self.calendar < date].max()
                    continue
                std_fairness = fairness_weight_value['std_fairness'].abs() * inday_return
                rtn_fairness = fairness_weight_value['rtn_fairness'].abs() * inday_return

                result_df = pd.concat([std_fairness.rename('波动公平因子'), rtn_fairness.rename('收益公平因子')], axis=1,
                                      join='inner')
                result_df = result_df.reset_index().rename(columns={'index': 'TICKER'})
                result_df['DATE'] = real_date
                feather.write_dataframe(result_df, os.path.join(self.tmp_path, f'{real_date}.feather'))
                break

        return result_df, pid

    def cal_min_data(self):
        """
        这一个part是利用分钟数据，计算日频因子的地方
        对一个因子来说，每只股票每天最后只对应有一个值
        """
        # 全历史日期序列
        output_path = os.path.join(self.tmp_path, '一视同仁中间数据.feather')
        if os.path.exists(output_path):
            df_old = feather.read_dataframe(output_path)
            max_date = df_old['DATE'].max()
        else:
            df_old = pd.DataFrame(columns=['TICKER', 'DATE'])
            max_date = self.start_date

        update_date_list = self.calendar[(self.calendar >= max_date) & (self.calendar <= self.end_date)]
        # if len(update_date_list) == 0:
        #     print('无需更新分钟因子')
        #     return
        # update_date_list = ['20250822']
        #
        # for date in update_date_list:
        #     res = self.minute_factor(date)

        result_list = Parallel(n_jobs=12)(delayed(self.minute_factor)(date) for date in
                                         tqdm(update_date_list, desc='Calculating Min Data'))
        pid_list = [x[1] for x in result_list]
        df_ls = [x[0] for x in result_list]
        kill_mutil_process(list(set(pid_list)))

        df_new = pd.concat(df_ls)
        df_old = pd.concat([df_old, df_new]).sort_values(by=['DATE', 'TICKER']).drop_duplicates()
        feather.write_dataframe(df_old, output_path)

    def cal_daily_factors(self):
        warnings.filterwarnings('ignore')
        basic_factor = feather.read_dataframe(os.path.join(self.tmp_path, f'一视同仁中间数据.feather'))
        basic_factor = basic_factor.sort_values(by=['TICKER', 'DATE'])
        basic_factor['波动公平因子_roll20'] = basic_factor.groupby('TICKER').apply(
            lambda x: x['波动公平因子'].rolling(20, min_periods=5).mean()).reset_index(level='TICKER', drop=True)
        basic_factor['收益公平因子_roll20'] = basic_factor.groupby('TICKER').apply(
            lambda x: x['收益公平因子'].rolling(20, min_periods=5).mean()).reset_index(level='TICKER', drop=True)

        basic_factor['一视同仁因子'] = basic_factor['波动公平因子_roll20'] + basic_factor['收益公平因子_roll20']
        # basic_factor['一视同仁因子_zscore'] = (basic_factor.groupby('DATE')['波动公平因子_roll20'].apply(zscore) + \
        #                                 basic_factor.groupby('DATE')['收益公平因子_roll20'].apply(zscore)).values

        # basic_factor['一视同仁因子_rank'] = basic_factor.groupby('DATE')['波动公平因子_roll20'].apply(lambda x:x.rank(pct=True)) + \
        #                                 basic_factor.groupby('DATE')['收益公平因子_roll20'].apply(lambda x:x.rank(pct=True))
        basic_factor = basic_factor.apply(lambda x: x.replace([np.inf, -np.inf], np.nan))
        # for col in basic_factor.columns.drop(['TICKER', 'DATE']):
        #     df_out = basic_factor[['TICKER', 'DATE', col]]
        #     print(df_out)
        #     feather.write_dataframe(df_out, os.path.join(self.output_path, col + '.feather'))
        return basic_factor

    def run(self):
        self.cal_min_data()
        df=self.cal_daily_factors()
        return df

def update(today='20250822'):
    if os.path.exists(os.path.join(DataPath.save_path_update,'波动公平因子_roll20.feather')):
        print('一视同仁因子更新中')
        old=feather.read_dataframe(os.path.join(DataPath.save_path_update,'波动公平因子_roll20.feather'))
        new_start=list(old.DATE.unique())[-50:][0]
        obj = No_Exception(start=new_start, end=today, out_path=DataPath.save_path_update)  # end=today
        df = obj.run()
        for col in ['波动公平因子_roll20','收益公平因子_roll20','一视同仁因子']:
            old=feather.read_dataframe(os.path.join(DataPath.save_path_update,col+'.feather'))
            df_out = df[['TICKER', 'DATE', col]]
            test_df=old.merge(df_out,on=['DATE','TICKER'],how='inner').dropna().tail(5)
            if np.isclose(test_df.iloc[:,2],test_df.iloc[:,3]).all():
                df_out=df_out[df_out.DATE>old.DATE.max()]
                df_out=pd.concat([old,df_out]).reset_index(drop=True)
                print(df_out)
                feather.write_dataframe(df_out, os.path.join(DataPath.save_path_update, col + '.feather'))
                feather.write_dataframe(df_out, os.path.join(DataPath.factor_out_path, col + '.feather'))
            else:
                print('数据检查更新有问题！')
                exit()
    else:
        print('一视同仁因子生成中')
        obj = No_Exception(start='20200101', end=today, out_path=DataPath.save_path_old)  # end=today
        df=obj.run()
        for col in ['波动公平因子_roll20','收益公平因子_roll20','一视同仁因子']:
            df_out = df[['TICKER', 'DATE', col]]
            print(df_out)
            # feather.write_dataframe(df_out, os.path.join(DataPath.save_path_old, col + '.feather'))
            feather.write_dataframe(df_out, os.path.join(DataPath.save_path_update, col + '.feather'))
            feather.write_dataframe(df_out, os.path.join(DataPath.factor_out_path, col + '.feather'))

if __name__ == '__main__':
    # today = dt.datetime.today().strftime('%Y%m%d')
    update('20250822')
    # obj = No_Exception(start='20250820', end='20250822', out_path=DataPath.save_path_update)  # end=today
    # df = obj.run()
    # print(df.groupby('DATE').count())
