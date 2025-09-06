import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
import os
from joblib import Parallel, delayed, cpu_count
import time
import datetime
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import kill_mutil_process
start = time.time()


class factor_withflow(object):
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 3000)
    warnings.filterwarnings(action='ignore')

    def __init__(self, start=None, end=None,savepath=None):
        self.start = start
        self.end = end
        self.path = DataPath.to_path
        calendar = pd.read_csv(self.path + '\\calendar.csv', dtype={'trade_date': str})
        calendar.rename(columns={'trade_date':'DATE'},inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']
        self.savepath =savepath
        self.tmp_path =DataPath.tmp_path
        self.calendar = calendar
        self.daily_path =DataPath.daily_path
        self.daily_basic_path =DataPath.to_df_path
        self.sz_min=DataPath.sz_min
        self.sh_min=DataPath.sh_min
        self.daily_list_file = os.listdir(self.sz_min)

    def get_daily_return_avg(self):
        # data_daily = pd.read_feather(self.daily_path, columns=['TICKER', 'DATE', 'adj_open', 'adj_close', 'amount'])
        data_daily = pd.read_feather(self.daily_path, columns=['TICKER', 'DATE', 'open', 'close', 'amount'])
        data_daily = data_daily[(data_daily['DATE'] <= self.end)&(data_daily['DATE'] >= self.start)]
        # data_daily = data_daily.pivot_table(columns=['TICKER'], index=['DATE'],
        #                                     values=['adj_open', 'adj_close', 'amount'])
        # data_daily = data_daily[(data_daily.index >= self.start) & (data_daily.index <= self.end)]
        # data_daily = data_daily.apply(lambda x: x.fillna(method='ffill'), axis=0)
        # data_daily = data_daily.stack(dropna=False).reset_index()

        if os.path.exists(self.tmp_path + '\\随波逐流'):
            # 检测下是否之前计算过该因子，如果有，就只再计算新多出来的这些天
            data_old = feather.read_dataframe(self.tmp_path + '\\随波逐流\\随波逐流.feather')
            cut_date = data_old['DATE'].drop_duplicates().sort_values().iloc[-40]
            data_old = data_old[data_old['DATE'] <= cut_date]
        else:
            data_old = pd.DataFrame(columns = ['TICKER','DATE'])
            cut_date = self.start
            os.makedirs(self.tmp_path + '\\随波逐流')

        # data_daily['日内收益率'] = data_daily['adj_close'] / data_daily['adj_open'] - 1
        data_daily['日内收益率'] = data_daily['close'] / data_daily['open'] - 1

        data_daily['20日合理收益率'] = data_daily.groupby('TICKER').apply(lambda x:
                     x['日内收益率'].rolling(20,min_periods=20).mean()).reset_index(level='TICKER', drop=True)

        data_daily = data_daily[['TICKER', 'DATE', 'amount', '20日合理收益率']]
        data_daily = data_daily[data_daily['DATE'] >= cut_date]

        data_daily = pd.concat([data_old, data_daily]).sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
        feather.write_dataframe(data_daily, self.tmp_path + r'\\随波逐流\\随波逐流.feather')

        data_daily[['高位成交额', '低位成交额', '高低差额']] = np.nan
        self.data_daily = data_daily.drop_duplicates()

    def cal_basic(self):
        if os.path.exists(self.tmp_path + r'\\随波逐流\\basic_min_data.feather'):
            df_old = feather.read_dataframe(self.tmp_path + r'\\随波逐流\\basic_min_data.feather')
            exist_date = [x.replace('-', '') for x in df_old['DATE'].unique()]
        else:
            df_old = pd.DataFrame()
            exist_date = []

        date_list = np.setdiff1d(self.daily_list, exist_date)

        def get_result(x, y):
            y.drop_duplicates(inplace=True)
            # x['min'] = [i.split(' ')[1] for i in x['trade_time'].tolist()]
            tick = x['TICKER'].drop_duplicates().values[0]
            x = x.drop(index=x[x['min'] == 930].index)
            open_value = x['open'][x['min'] ==931].values
            if len(open_value) != 0:
                open_value = open_value
            else:
                return None
            if tick not in y.index:
                return None
            x['min_relative_return'] = x['close'] / open_value - 1
            big = (x['amount'][x['min_relative_return'] > y.loc[tick]['20日合理收益率']]).sum()
            small = (x['amount'][x['min_relative_return'] < y.loc[tick]['20日合理收益率']]).sum()

            return pd.Series([big, small], index=['高位成交额', '低位成交额'])

        def cal_daily(date):
            warnings.filterwarnings('ignore')
            pid = os.getpid()
            # filename = '-'.join([date[:4], date[4:6], date[6:8]]) + '.feather'
            filename=date+'.feather'
            sz_min,sh_min=pd.DataFrame(),pd.DataFrame()
            try:
                sz_min = feather.read_dataframe(os.path.join(self.sz_min,filename))
            except FileNotFoundError:
                pass
            try:
                sh_min = feather.read_dataframe(os.path.join(self.sh_min,filename))
            except FileNotFoundError:
                pass
            min=pd.concat([sz_min,sh_min]).reset_index(drop=True)
            if min.empty:
                return
            daily_df = self.data_daily[self.data_daily['DATE'] == date].set_index(['TICKER'])[['20日合理收益率','amount']].drop_duplicates()
            daily_df.dropna(subset=['amount'])

            if len(daily_df) == 0:
                df = min[['TICKER']]
                df['DATE'] = date

            else:
                df = min.groupby('TICKER').apply(get_result, y=daily_df)
                df = df.reset_index()
                df['DATE'] = date
            return df, pid

        if len(date_list) > 0:
            result_ls = Parallel(n_jobs=12)(delayed(cal_daily)(date) for date in tqdm(date_list, desc='cal'))
            result_ls=[x for x in result_ls if x is not None]
            df_list = [x[0] for x in result_ls]
            pid_list = [x[1] for x in result_ls]
            kill_mutil_process(pid_list)

            df_new = pd.DataFrame()._append(df_list)

            df_old = pd.concat([df_old, df_new]).sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
            feather.write_dataframe(df_old, self.tmp_path + r'\\随波逐流\\basic_min_data.feather')
        return df_old
    def cal(self):
        if os.path.exists(self.savepath + f'\\随波逐流.feather'):
            dz = pd.read_feather(self.savepath + f'\\随波逐流.feather')
            max_date = dz['DATE'].max()
        else:
            dz = pd.DataFrame()
            max_date = '20130104'
        # try:
        #     to_day = feather.read_dataframe(self.tmp_path + r'\\随波逐流\\basic_min_data.feather')
        # except FileNotFoundError:
        to_day = self.cal_basic()
        to_day['高低差额'] = (to_day['高位成交额'] - to_day['低位成交额']) / (
                to_day['高位成交额'] + to_day['低位成交额'])

        table = pd.pivot_table(to_day, values='高低差额', index='DATE', columns='TICKER')
        table.fillna(method='ffill')

        def cal_spear(i):
            pid = os.getpid()
            table_temp = table.iloc[i - 19:i + 1, :].dropna(how='any', axis=1)
            spear_daily = table_temp.rank().corr().abs().mean().reset_index(name = '随波逐流')
            spear_daily['DATE'] = table.index[i]
            return spear_daily, pid

        index_list = [x for x in range(19, len(table)) if table.index[x] > max_date]
        if len(index_list) > 0:
            result_ls = Parallel(n_jobs=12)(delayed(cal_spear)(i) for i in tqdm(index_list, desc='正在计算spear系数'))
            result_ls = [x for x in result_ls if x is not None]
            spear_all_ls = [x[0] for x in result_ls]
            pid_list = [x[1] for x in result_ls]
            kill_mutil_process(list(set(pid_list)))
            spear_all = pd.DataFrame()._append(spear_all_ls)

            spear_all = spear_all.rename({'level_0': 'TICKER'}, axis=1)
            spear_all = spear_all[['TICKER', 'DATE','随波逐流']]
            # data_daily = feather.read_dataframe(self.daily_path)
            # data_daily = data_daily[(data_daily['DATE'] >= self.start) & (data_daily['DATE'] <= self.end)]
            # spear_all = data_daily.merge(spear_all, how='left')
            nowdata = pd.concat([dz, spear_all])
            # nowdata = spear_all[['TICKER', 'DATE'] + [Fct_list[i]]]
            # feather.write_dataframe(nowdata, self.savepath + f'\\随波逐流.feather')
            # feather.write_dataframe(nowdata, DataPath.save_path_update + f'\\随波逐流.feather')
        # print(nowdata)
        # print(f'随波逐流 has been saved')
            return nowdata

    def run(self):
        self.get_daily_return_avg()
        # self.cal_basic()
        df=self.cal()
        return df

def update(today='20250822'):
    # today = datetime.datetime.today().strftime('%Y%m%d')
    if os.path.exists(os.path.join(DataPath.save_path_update, '随波逐流.feather')):
        print('随波逐流因子更新中')
        old = feather.read_dataframe(os.path.join(DataPath.save_path_update, '随波逐流.feather')).drop_duplicates()
        tar_dl = sorted(list(old.DATE.unique()))[-50:][0]
        object = factor_withflow(start=tar_dl, end=today,savepath=DataPath.save_path_update)
        # df1,df2,df3,df4=object.cal()
        df = object.run()
        test = old.merge(df, how='inner', on=['DATE', 'TICKER']).dropna()#.tail(5)
        test.sort_values(['TICKER','DATE'],inplace=True)
        test=test.groupby('TICKER').tail(5).drop_duplicates()
        if np.isclose(test.iloc[:, 2], test.iloc[:, 3]).all():
            df=pd.concat([old,df]).reset_index(drop=True).drop_duplicates()
            print(df)
            feather.write_dataframe(df, os.path.join(DataPath.save_path_update, '随波逐流.feather'))
            feather.write_dataframe(df, os.path.join(DataPath.factor_out_path, '随波逐流.feather'))
        else:
            print('数据检查更新出错')
            exit()
    else:
        print('随波逐流因子生成中')
        object = factor_withflow(start='20200101', end=today,savepath=DataPath.save_path_old)
        # break_point = 1
        df = object.run()
        df.drop_duplicates(inplace=True)
        feather.write_dataframe(df, os.path.join(DataPath.save_path_old, '随波逐流.feather'))
        feather.write_dataframe(df, os.path.join(DataPath.save_path_update, '随波逐流.feather'))
#

if __name__ == '__main__':
    update()
