import os

import feather
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from tqdm import tqdm

from tool_tyx.path_data import DataPath

def gen_ideal_v(window_size,ratio,tmp_df):
    # tmp_date = tar_date[i - (window_size - 1): i + 1]  # 取前20天（含当前）
    # tmp_df = df[df['DATE'].isin(tmp_date)]
    tmp_df.sort_values(['TICKER', 'close'], inplace=True)
    ideal_v_high = tmp_df.groupby('TICKER').agg(
        ideal_v_high=('amplitude', lambda x: x.head(int(window_size * ratio)).mean()))
    ideal_v_low = tmp_df.groupby('TICKER').agg(
        ideal_v_low=('amplitude', lambda x: x.tail(int(window_size * ratio)).mean()))
    ideal_v = ideal_v_low.merge(ideal_v_high, left_index=True, right_index=True).reset_index()
    ideal_v['ideal_v_diff'] = ideal_v['ideal_v_high'] - ideal_v['ideal_v_low']
    ideal_v['DATE']=tmp_df.DATE.max()
    return ideal_v

def read_data(start='20250101',end='20250331',df_path=DataPath.daily_path,window_size=20,ratio=0.25):
    warnings.filterwarnings('ignore')
    df=feather.read_dataframe(df_path)[['DATE','TICKER','high','low','close']]
    df=df[(df.DATE>=start)&(df.DATE<=end)]
    df['amplitude']=df['high']/df['low']-1
    df.sort_values(['TICKER','DATE'],inplace=True)
    tar_date=sorted(df.DATE.unique())
    res=Parallel(n_jobs=10)(delayed(gen_ideal_v)(window_size,ratio,df[df['DATE'].isin(tar_date[i - (window_size - 1): i + 1])]) for i in tqdm(range(window_size-1, len(tar_date))))
    res=pd.concat(res).reset_index(drop=True)
    df1=res[['DATE','TICKER','ideal_v_high']]
    df2=res[['DATE','TICKER','ideal_v_low']]
    df3=res[['DATE','TICKER','ideal_v_diff']]
    return df1,df2,df3

def update(today='20250820'):
    save_path_old = DataPath.save_path_old
    save_path_update = DataPath.save_path_update
    flag = True
    while flag:
        if os.path.exists(os.path.join(save_path_update, 'ideal_v_diff.feather')):
            old_df = feather.read_dataframe(os.path.join(save_path_update, 'ideal_v_diff.feather'))
            old_date_list = old_df.DATE.drop_duplicates().to_list()
            old_date_list = sorted(old_date_list)
            new_start = old_date_list[old_date_list.index(old_df.DATE.max()) - 30]
            print(f'理想振幅系列因子更新中：')
            df1, df2, df3=read_data(start=new_start,end=today)  # todo: 每次更新检查，改end
            print('------------------------')
            for name,data in {'ideal_v_high':df1,'ideal_v_low':df2,'ideal_v_diff':df3}.items():
                old_df=feather.read_dataframe(os.path.join(save_path_update,name+'.feather'))
                test_df = old_df.merge(data, on=['DATE', 'TICKER'], how='inner').dropna()
                # print(test_df)
                if np.isclose(test_df.iloc[:, 2], test_df.iloc[:, 3]).all():
                    new_df = data[data.DATE > old_df.DATE.max()]
                    full_df = pd.concat([old_df, new_df]).reset_index(drop=True)
                    feather.write_dataframe(full_df, os.path.join(save_path_update, name+'.feather'))
                    feather.write_dataframe(full_df, os.path.join(DataPath.factor_out_path, name+'.feather'))
                else:
                    print('检查更新，数据有问题！')
                    exit()
            flag = False
        else:
            print(f'理想振幅因子生成中：')
            df1,df2,df3=read_data(start='20200101',end='20221231')# 第一次生成因子
            for name,data in {'ideal_v_high':df1,'ideal_v_low':df2,'ideal_v_diff':df3}.items():
                feather.write_dataframe(data, os.path.join(save_path_old, name+'.feather'))
                feather.write_dataframe(data, os.path.join(save_path_update,name+'.feather'))
if __name__=='__main__':
    # df1,df2,df3=read_data(start='20200101',end='20221231')
    #
    # df1,df2,df3=read_data(start='20210601',end='20250801')
    update()
