import os
import feather
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
from tqdm import tqdm

from tool_tyx.tyx_funcs import process_na_stock
from tool_tyx.path_data import DataPath


#对于每只股票的子df,这里的df_sub是经过TICKER和DATE排序并索引重置后的结果，计算日频GTR
def cal_GTR(df_sub):
    df_sub['GTR']=df_sub['turnover_rate'].rolling(window=20).std()
    #print(df_sub)
    return df_sub[['DATE','TICKER','GTR']]

class GTR:
    def __init__(self, end=None, data_path=None, output_file_directory=None):
        self.end_date = end or '20230301'
        self.data_path = DataPath.to_df_path
        # self.save_path_old = DataPath.save_path_old
        # os.makedirs(self.output_file_directory, exist_ok=True)

    def gen_GTR(self):
        warnings.filterwarnings('ignore')
        df_daily = feather.read_dataframe(os.path.join(self.data_path, 'daily.feather'),columns = ['TICKER',
                                                                                                   'DATE','volume'])
        df_float_mv=feather.read_dataframe(os.path.join(self.data_path,'float_mv.feather'))
        df_daily = pd.merge(df_daily, df_float_mv, on=['TICKER', 'DATE'], how='left')
        df_daily['free_turnover']=df_daily['volume']/df_daily['float_mv']
        df_daily=df_daily[['TICKER','DATE','free_turnover']]

        print('data loaded')
        df_daily = df_daily[df_daily['DATE'] <= self.end_date].sort_values(by=['TICKER', 'DATE'])

        turnover_rate = df_daily.groupby('TICKER').apply(lambda x:x['free_turnover'] / x['free_turnover'].shift(1) - 1)
        df_daily['turnover_rate'] = turnover_rate.reset_index(level = 'TICKER', drop = True)
        df_daily['turnover_rate'] = df_daily['turnover_rate'].replace([np.inf, -np.inf], np.nan)
        df_daily.replace([np.inf,-np.inf],np.nan,inplace=True)
        df_daily=process_na_stock(df_daily,'turnover_rate')
        GTR = df_daily.groupby('TICKER').apply(lambda x:x['turnover_rate'].rolling(window=20,min_periods=5).std())

        df_daily['GTR'] = GTR.reset_index(level = 'TICKER', drop = True)
        df_daily['GTR'] = df_daily['GTR'].replace([np.inf, -np.inf], np.nan)
        # feather.write_dataframe(df_daily[['TICKER','DATE','GTR']],
        #                         os.path.join(self.save_path_old, 'GTR.feather'))
        return df_daily[['TICKER','DATE','GTR']]
        # print(df_daily)
    def run(self):
        data=self.gen_GTR()
        return data

def update(today='20250904'):
    save_path_old = DataPath.save_path_old
    save_path_update = DataPath.save_path_update
    flag = True
    while flag:
        if os.path.exists(os.path.join(save_path_update, 'GTR.feather')):
            old_df = feather.read_dataframe(os.path.join(save_path_update, 'GTR.feather'))
            old_date_list = old_df.DATE.drop_duplicates().to_list()
            old_date_list = sorted(old_date_list)
            # new_start = old_date_list[old_date_list.index(old_df.DATE.max()) - 50]
            print(f'因子{'GTR.feather'.replace('.feather', '')}更新中：')
            data = GTR(end=today).run()  # 每次更新检查，改end
            print('------------------------')
            test_df = old_df.merge(data, on=['DATE', 'TICKER'], how='inner').dropna()#.tail(5)
            test_df.sort_values(['TICKER','DATE'],inplace=True)
            tar_list = sorted(list(test_df.DATE.unique()))[-5:]
            test_df = test_df[test_df.DATE.isin(tar_list)]
            # print(test_df)
            if np.isclose(test_df.iloc[:, 2], test_df.iloc[:, 3]).all():
                new_df = data[data.DATE > old_df.DATE.max()]
                full_df = pd.concat([old_df, new_df]).reset_index(drop=True)
                print(full_df)
                feather.write_dataframe(full_df, os.path.join(save_path_update, 'GTR.feather'))
                feather.write_dataframe(full_df,os.path.join(DataPath.factor_out_path,'GTR.feather'))
            else:
                print(test_df[~np.isclose(test_df.iloc[:, 2], test_df.iloc[:, 3])])
                print('检查更新，数据有问题！')
                exit()
            flag = False
        else:
            print(f'新因子{'GTR_factor.feather'.replace('.feather', '')}生成中：')
            data = GTR(end='20251021').run()  # 第一次生成因子
            print(data)
            feather.write_dataframe(data, os.path.join(DataPath.factor_out_path, 'GTR.feather'))
            feather.write_dataframe(data, os.path.join(save_path_old, 'GTR.feather'))
            feather.write_dataframe(data, os.path.join(save_path_update, 'GTR.feather'))

if __name__ == '__main__':
    update('20251021')









