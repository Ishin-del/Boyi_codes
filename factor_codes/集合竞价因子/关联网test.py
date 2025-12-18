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


def process_na_stock(df,col):
    '''解决某些股票出现时间序列断开的情况，这种情况会在rolling（window!=min_period）的时候,因子值出现不同'''
    '''重组再上市情况'''
    pt = df.pivot_table(columns='TICKER', index='DATE', values=col)
    pt = pt.ffill()
    pp = pt.melt(ignore_index=False).reset_index(drop=False)
    df = pd.merge(pp, df, how='left', on=['TICKER', 'DATE'])
    use_cols = list(np.setdiff1d(df.columns, ['TICKER', 'DATE']))
    for c in use_cols:
        df[c] = df.groupby('TICKER')[c].ffill()
    df.drop(columns='value', inplace=True)
    return df


def process_na_auc(df,col):
    '''解决某些股票出现时间序列断开的情况，这种情况会在rolling（window!=min_period）的时候,因子值出现不同'''
    '''重组再上市情况'''
    pt = df.pivot_table(columns='TICKER', index='time', values=col)
    pt = pt.ffill()
    pp = pt.melt(ignore_index=False).reset_index(drop=False)
    df = pd.merge(pp, df, how='left', on=['TICKER', 'time'])
    use_cols = list(np.setdiff1d(df.columns, ['TICKER', 'time']))
    for c in use_cols:
        df[c] = df.groupby('TICKER')[c].ffill()
    df.drop(columns='value', inplace=True)
    return df

class Traction_LUD:
    warnings.filterwarnings('ignore')
    def __init__(self, start, end, calender_path=None, df_path=None, savepath=None, daily_path=None,split_sec=1):
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

        self.daily_path = daily_path or r'Z:\local_data\Data_Storage\daily.feather'

        self.daily_df = feather.read_dataframe(self.daily_path)[['DATE', 'TICKER', 'close']]
        self.daily_df.sort_values(['TICKER','DATE'],inplace=True)
        self.daily_df['close']=self.daily_df.groupby('TICKER')['close'].shift(1) # close数据用前一天的
        self.daily_df['pre_close']=self.daily_df['close']
        self.daily_df = self.daily_df[self.daily_df['DATE'] > '20191229']
        self.daily_df.sort_values(['TICKER', 'DATE'], inplace=True)
        self.daily_df = process_na_stock(self.daily_df, 'close')
        self.daily_df['close_roll20'] = self.daily_df.groupby('TICKER')['close'].shift(20)
        self.daily_df['ret_roll20'] = self.daily_df['close'] / self.daily_df['close_roll20'] - 1
        self.daily_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # self.daily_df = self.daily_df[self.daily_df['DATE'].isin(self.daily_list)]

    def __daily_calculation(self):
        if os.path.exists(os.path.join(self.tmp_path, 'traction_lud.feather')):
            df_old = feather.read_dataframe(os.path.join(self.tmp_path, 'traction_lud.feather'))
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []
        max_min_date = max(os.listdir(self.df_path)).replace('.feather', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)
        date_list=[x for x in date_list if x not in ['20240206', '20240207', '20240208', '20240926', '20240927',
                                                     '20240930', '20241008']]
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
                dt.rename(columns={'volume_diff': 'volume'}, inplace=True)
                dt.sort_values(['TICKER','time'],inplace=True)
                dt=process_na_auc(dt,'close')
                """
                这里用前面的数据填充了一下集合竞价数据中的缺失值（不存在未来数据），否则，因为后面代码中的点乘，最终因子值缺失值会太多
                """
                # -----------------------------------------------------------
                ret_df=self.daily_df[self.daily_df['DATE']==date][['TICKER','ret_roll20']] #,'pre_close'
                dt = dt.merge(ret_df, on='TICKER', how='inner')
                dt.sort_values(['TICKER','time'],inplace=True)
                # todo:
                # dt['time_ret'] = dt.groupby('TICKER')['close'].pct_change()
                dt['time_volume_change'] = dt.groupby('TICKER')['volume'].pct_change()
                auc_close=dt.groupby('TICKER')[['TICKER','close']].tail(1)
                auc_open=dt.groupby('TICKER')[['TICKER','open']].head(1)
                pre_ret=auc_open.merge(auc_close,on='TICKER',how='inner')
                pre_ret['pre_ret']=pre_ret['close']/pre_ret['open']-1
                pre_ret=pre_ret[['TICKER','pre_ret']]
                # todo:
                dt['dir_up'] = np.where(dt['time_volume_change'] > 0, 1, 0)
                dt['dir_down'] = np.where(dt['time_volume_change'] < 0, 1, 0)

                useA = dt[['TICKER', 'time', 'dir_up']].pivot(index='time', columns='TICKER', values='dir_up')
                useB = dt[['TICKER', 'time', 'dir_down']].pivot(index='time', columns='TICKER', values='dir_down')
                # 分别计算同时上涨和同时下跌的比例
                up_res = np.dot(useA.T.values, useA.values)
                down_res = np.dot(useB.T.values, useB.values)
                result_up = up_res / len(useA)
                result_down = down_res / len(useB)
                result_df1 = pd.DataFrame(result_up, index=useA.T.index, columns=useA.columns).T
                result_df2 = pd.DataFrame(result_down, index=useB.T.index, columns=useB.columns).T
                # 同时上涨和同时下跌的比例加起来，得到最终同向比例
                result_df = result_df1 + result_df2
                # --------------------------------------
                # 寻找同向比例的最大50%
                num = int(len(result_df)*0.75)+1
                # num=41
                tmp_values = result_df.apply(lambda x: x.nlargest(num).values).tail(num-1)
                tmp_codes = result_df.apply(lambda x: x.nlargest(num).index).tail(num-1)
                # 得到最大50%股票的ret，跟前面的同向比例计算加权平均，得到股票的预期收益
                tmp_dict = ret_df.set_index('TICKER')['ret_roll20'].to_dict()
                tmp_ret = tmp_codes.applymap(lambda x: tmp_dict.get(x))
                res = (tmp_values * tmp_ret).sum() / tmp_values.sum()
                res = pd.DataFrame(res).reset_index()
                res['DATE'] = date
                res = res[['DATE', 'TICKER', 0]].rename(columns={0: 'exp_LUD'})
                res=res.merge(pre_ret,on='TICKER',how='inner')
                res['traction_LUD ']=res['pre_ret']-res['exp_LUD']
                res.drop(columns='pre_ret',inplace=True)
                return res
        get_daily('20220128')
        # df = Parallel(n_jobs=10)(delayed(get_daily)(date) for date in tqdm(date_list))
        # df = pd.concat(df).reset_index(drop=True)
        # alls = pd.concat([df_old, df]).reset_index(drop=True)
        # feather.write_dataframe(alls, os.path.join(self.tmp_path, 'traction_lud.feather'))
        # print(alls)

    def cal(self):
        warnings.filterwarnings('ignore')
        df = feather.read_dataframe(os.path.join(self.tmp_path, 'traction_lud.feather'))
        tar_col = list(np.setdiff1d(df.columns, ['TICKER', 'DATE']))
        for c in tar_col:
            tmp = df[['TICKER', 'DATE', c]]
            tmp.rename(columns={c: 'auc_' + c}, inplace=True)
            c = 'auc_' + c
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp[c] = tmp[c].fillna(tmp.groupby('DATE')[c].transform('median'))
            feather.write_dataframe(tmp, os.path.join(self.savepath, c + '.feather'))
            # print(tmp)

    def run(self):
        self.__daily_calculation()
        # self.cal()

if __name__=='__main__':
    obj=Traction_LUD(start='20220101',end='20241231',
        df_path=r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time\concat_daily',
                       savepath=r'C:\Users\admin\Desktop',
                       split_sec=5)
    obj.run()