import datetime
import os
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib.parallel import Parallel, delayed, cpu_count
import warnings
import math
import time

from tool_tyx.path_data import DataPath
class factor_leftail(object):
    pd.set_option('display.max_columns', 18)
    pd.set_option('display.width', 1000)
    warnings.filterwarnings(action='ignore')

    def __init__(self, start, end, window):
        self.start = start
        self.end = end
        self.path = DataPath.to_path
        # self.savepath = DataPath.out_path
        calendar = pd.read_csv(self.path + '\\calendar.csv', dtype={'trade_date': str})
        self.calendar = calendar
        self.daily_path = DataPath.daily_path
        self.daily_basic_path = os.path.join(DataPath.to_df_path,'daily_basic.feather')
        self.window=window

    def cal(self):
        data = feather.read_dataframe(self.daily_path, columns=['TICKER', 'DATE', 'close'])
        data.sort_values(['TICKER','DATE'],inplace=True)
        # data['pct_chg'] = (data['adj_close'] / data['adj_pre_close'] - 1 ) * 100
        data['pct_chg'] =data.groupby('TICKER').close.pct_change() * 100
        data = data[(data['DATE']>=self.start)&(data['DATE'] <= self.end)].reset_index(drop=True)
        # data = data.pivot_table(columns=['TICKER'], index=['DATE'],
        #                         values=['pct_chg'])
        # data = data.stack(dropna=False).reset_index()
        # if os.path.exists(self.savepath + r'\VaR_5%.feather'):
        #     data_old = feather.read_dataframe(self.savepath + r'\VaR_5%.feather')
        #     dz = data_old[data_old['DATE'] <= self.end]
        #     dtls = dz['DATE'].unique()
        #     dtls.sort()
        #     data = data[data['DATE'] > dtls[(max(len(dtls) - 300, 0))]]

        def func(x):
            x['VaR_5%'] = x['pct_chg'].rolling(self.window, min_periods=self.window).mean() + x['pct_chg'].rolling(self.window,min_periods=self.window).std() * 1.65
            x['VaR_1%'] = x['pct_chg'].rolling(self.window, min_periods=self.window).mean() + x['pct_chg'].rolling(self.window,min_periods=self.window).std() * 2.33
            x['ES_5%'] = x['pct_chg'].rolling(self.window, min_periods=self.window).mean() + x['pct_chg'].rolling(self.window, min_periods=self.window).std() * (math.exp(-(1.65 ** 2 / 2))) / (np.sqrt(2 * math.pi) * 0.05)
            x['ES_1%'] = x['pct_chg'].rolling(self.window, min_periods=self.window).mean() + x['pct_chg'].rolling(self.window, min_periods=self.window).std() * (math.exp(-(2.33 ** 2 / 2))) / (np.sqrt(2 * math.pi) * 0.01)
            return x

        tqdm.pandas(desc='leftail')
        factor_df = data.groupby('TICKER').progress_apply(func)#
        factor_df.reset_index(drop=True,inplace=True)
        # Fct_list = ['ES_1%', 'ES_5%', 'VaR_1%', 'VaR_5%']
        Fct_list = ['VaR_5%']
        # for i in range(len(Fct_list)):
        #     nowdata = factor_df[['TICKER', 'DATE'] + [Fct_list[i]]]
        #     print(nowdata)
            # feather.write_dataframe(nowdata, self.savepath + f'{Fct_list[i]}.feather')
            # print(f'{Fct_list[i]} has been saved')
        df1=factor_df[['DATE','TICKER',Fct_list[0]]]
        df1.rename(columns={'VaR_5%':'VaR_5%_v2'},inplace=True)
        # df2=factor_df[['DATE','TICKER',Fct_list[1]]]
        # df3=factor_df[['DATE','TICKER',Fct_list[2]]]
        # df4=factor_df[['DATE','TICKER',Fct_list[3]]]
        return df1
    #
    # def run(self):
    #     self.cal()

def update(today='20250820'):
    # today = datetime.datetime.today().strftime('%Y%m%d')
    print('右尾风险因子更新中')
    if os.path.exists(os.path.join(DataPath.save_path_update,'VaR_5%_v2.feather')):
        old = feather.read_dataframe(os.path.join(DataPath.save_path_update, 'VaR_5%_v2.feather'))
        tar_dl = sorted(list(old.DATE.unique()))[-50:][0]
        object = factor_leftail(start=tar_dl, end=today, window=20)
        df4=object.cal()
        for file_name,data in {'VaR_5%_v2':df4}.items():
            file_name+='.feather'
            old=feather.read_dataframe(os.path.join(DataPath.save_path_update,file_name))
            test=old.merge(data,how='inner',on=['DATE','TICKER']).dropna()#.tail(5)
            tar_list = sorted(list(test.DATE.unique()))[-5:]
            test = test[test.DATE.isin(tar_list)]
            if np.isclose(test.iloc[:,2],test.iloc[:,3]).all():
                data = data[data.DATE > old.DATE.max()]
                data = pd.concat([old, data]).reset_index(drop=True)
                print(data)
                feather.write_dataframe(data, os.path.join(DataPath.save_path_update,file_name))
                feather.write_dataframe(data, os.path.join(DataPath.factor_out_path,file_name))
            else:
                print('数据检查更新出错')
                exit()
    else:
        print('右尾风险因子生成中')
        object = factor_leftail(start='20200101', end=today, window=20)
        # break_point = 1
        df4=object.cal()
        # print("运行时间：", time.time() - start)
        for file_name,data in {'VaR_5%_v2':df4}.items():
            file_name+='.feather'
            print(data)
            feather.write_dataframe(data, os.path.join(DataPath.save_path_old,file_name))
            feather.write_dataframe(data, os.path.join(DataPath.save_path_update,file_name))
            feather.write_dataframe(data, os.path.join(DataPath.factor_out_path, file_name))

if __name__ == '__main__':
    update()
