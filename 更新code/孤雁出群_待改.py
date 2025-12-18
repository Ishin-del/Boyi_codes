import feather
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from tqdm import tqdm
import warnings
import os
import time

from tool_tyx.path_data import DataPath

start = time.time()


class factor_outside(object):
    # pd.set_option('display.max_columns', 20)
    # pd.set_option('display.width', 200)
    warnings.filterwarnings(action='ignore')

    def __init__(self, start=None, end=None, savepath=None):
        self.start = start
        self.end = end
        self.path = DataPath.to_path
        self.mkt_return_path = DataPath.wind_A_path
        self.start = start
        self.end = end
        calendar = pd.read_csv(self.path + '\\calendar.csv', dtype={'trade_date': str})
        calendar.rename(columns={'trade_date':'DATE'},inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']
        self.savepath = savepath
        self.tmp_path=DataPath.tmp_path
        self.calendar = calendar
        self.daily_path = DataPath.daily_path
        self.daily_basic_path = os.path.join(DataPath.to_df_path,'daily_basic.feather')
        self.sh_min = DataPath.sh_min
        self.sz_min = DataPath.sz_min
        self.daily_list_file = os.listdir(self.sh_min)

    def helper(self,date):
        warnings.filterwarnings(action='ignore')
        # filename = '-'.join([date[:4], date[4:6], date[6:8]]) + '.feather'
        filename = date + '.feather'
        sh_min, sz_min = pd.DataFrame(), pd.DataFrame()
        try:
            sh_min = feather.read_dataframe(os.path.join(self.sh_min, filename))
        except FileNotFoundError:
            pass
        try:
            sz_min = feather.read_dataframe(os.path.join(self.sz_min, filename))
        except FileNotFoundError:
            pass
        min = pd.concat([sh_min, sz_min]).reset_index(drop=True)
        if min.empty:
            return
        # min['min'] = [i.split(' ')[1] for i in min['trade_time'].tolist()]
        # min = min.drop(index=min[min['min'] == 930].index)
        # min = min.drop(index=min[min['min'] == 1458].index)
        # min = min.drop(index=min[min['min'] == 1459].index)
        # min = min.drop(index=min[min['min'] == 1500].index)
        min=min[min['min']>930]
        min=min[(min['min']<1457)|(min['min']>1459)].drop_duplicates()
        min.sort_values(['TICKER','min'],inplace=True)
        min = min.reset_index(drop=True)
        # min['pre_min_close']=min.groupby('TICKER').close.shift(1)
        # min['change_min'] = min['close'] / min['pre_min_close'] - 1
        min['change_min'] = min.groupby('TICKER')['close'].pct_change()
        # min['change_min'] = min['close'] / min['open'] - 1
        min['change_min'] = min['change_min'].replace([np.inf,-np.inf],np.nan)
        min_std = min[['min', 'change_min']].groupby('min').apply(lambda x:np.nanstd(x['change_min'])) #市场分化度
        min_nodivided = min_std[(min_std < min_std.mean())]
        min_nodivided = min[['TICKER', 'min', 'change_min', 'amount']][
            min['min'].isin(min_nodivided.index)]
        daily_nodivided = min_nodivided.pivot_table(columns=['TICKER'], index=['min'],
                                                    values=['amount'])
        # daily_nodivided_corr = daily_nodivided.corr()
        # daily_nodivided_corr_values_rank = daily_nodivided_corr.rank(axis=1,ascending=False)
        # daily_nodivided_corr_values_rank = daily_nodivided_corr_values_rank  < 31
        # daily_nodivided_corr = daily_nodivided_corr*daily_nodivided_corr_values_rank


        daily_nodivided = (daily_nodivided.corr().abs().sum(axis=1) - 1)/daily_nodivided.shape[1]
        daily_nodivided = pd.DataFrame(daily_nodivided.rename('孤雁出群'))
        daily_nodivided = daily_nodivided.reset_index(level='TICKER')
        daily_nodivided.reset_index(drop=True,inplace=True)
        daily_nodivided['DATE'] = str(date)
        return daily_nodivided

    def cal(self):
        daily_nodivided_all = pd.DataFrame()
        if os.path.exists(self.tmp_path + r'\孤雁出群.feather'):
            # 检测下是否之前计算过该因子，如果有，就只再计算新多出来的这些天
            df_old = feather.read_dataframe(self.tmp_path + r'\孤雁出群.feather')
            exist_date = [x for x in df_old['DATE'].unique()]
        else:
            df_old = pd.DataFrame()
            exist_date = []
        # max_min_date = max(self.daily_list_file)[:10].replace('-', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)

        # daily_nodivided = []
        # for date in tqdm(date_list, desc='calculating daily'):
        #     daily_nodivided.append(self.helper(date))

        daily_nodivided=Parallel(n_jobs=1)(delayed(self.helper)(date) for date in tqdm(date_list, desc='calculating daily'))
        daily_nodivided=pd.concat(daily_nodivided).reset_index(drop=True)
        daily_nodivided_all = pd.concat([daily_nodivided_all, daily_nodivided])
        if date_list.any():
            daily_nodivided_all = daily_nodivided_all.reset_index(drop=True)
            df_old = pd.concat([df_old, daily_nodivided_all])
            feather.write_dataframe(df_old, self.tmp_path + f'\\孤雁出群.feather')
            print(f'孤雁出群 has been saved')
        # return df_old

    def month(self):
        data_gu = feather.read_dataframe(self.tmp_path + '\\孤雁出群.feather')
        data_gu = data_gu[(data_gu['DATE'] >= self.start) & (data_gu['DATE'] <= self.end)]
        data_gu['20日均孤雁出群'] = data_gu.groupby('TICKER').apply(lambda x:
                                 x['孤雁出群'].rolling(20, min_periods=20).mean()).reset_index(level = 'TICKER',drop=True).values
        data_gu['20日稳孤雁出群'] = data_gu.groupby('TICKER').apply(lambda x:
                                 x['孤雁出群'].rolling(20, min_periods=20).std()).reset_index(level = 'TICKER',drop=True).values

        data_gu['20日孤雁出群'] = (data_gu['20日稳孤雁出群'] + data_gu['20日均孤雁出群'])/2
        Fct_list = ['20日稳孤雁出群', '20日均孤雁出群', '20日孤雁出群']
        for i in range(len(Fct_list)):
            nowdata = data_gu[['TICKER', 'DATE'] + [Fct_list[i]]]
            if os.path.exists(os.path.join(self.savepath,f'{Fct_list[i]}.feather')):
                old=feather.read_dataframe(os.path.join(self.savepath,f'{Fct_list[i]}.feather'))
                nowdata=nowdata[nowdata.DATE>old.DATE.max()]
                old=pd.concat([old,nowdata]).reset_index(drop=True).drop_duplicates()
                print(old)
                feather.write_dataframe(old, self.savepath + f'\\{Fct_list[i]}.feather')
                # feather.write_dataframe(old, self.savepath + f'\\{Fct_list[i]}.feather')
            else:
                feather.write_dataframe(nowdata, self.savepath + f'\\{Fct_list[i]}.feather')
                feather.write_dataframe(nowdata, DataPath.save_path_update + f'\\{Fct_list[i]}.feather')

    def run(self):
        self.cal()
        # self.month()

def update(today='20250820'):
    object = factor_outside(start='20200101', end=today, savepath=DataPath.save_path_old)
    object.run()


if __name__ == '__main__':
    update()
