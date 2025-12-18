import os
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings
from joblib import Parallel,delayed
from tool_tyx.path_data import DataPath

def read_data(start='',end='',df_path=DataPath.daily_path):
    warnings.filterwarnings('ignore')
    df=feather.read_dataframe(df_path,columns=['TICKER', 'DATE', 'volume'])
    # test:
    # df=df[df.TICKER=='000096.SZ']
    df=df[(df['DATE'] <= end) & (df['DATE'] >= start)]
    tar_ticker = df.TICKER.drop_duplicates()
    tar_date = df.DATE.drop_duplicates()
    return tar_ticker,tar_date,df[['TICKER','DATE']].drop_duplicates()

def cal_data(file,path,tar_ticker):
    warnings.filterwarnings('ignore')
    try:
        tmp = feather.read_dataframe(os.path.join(path, file+'.feather'),columns=['TICKER', 'min','volume','DATE'])
    except FileNotFoundError:
        return
    tmp = tmp[tmp.TICKER.isin(tar_ticker)]
    """分钟数据的volume的均值和标准差"""
    tmp = tmp[(tmp['min'] < 1457) | (tmp['min'] > 1459)]
    tmp=tmp.groupby(['TICKER','DATE'])['volume'].agg(['std','mean']).reset_index()
    tmp['vcv_daily']=tmp['std']/tmp['mean']
    tmp.drop(columns=['std', 'mean'],inplace=True)
    tmp=tmp[['TICKER', 'DATE','vcv_daily']]
    return tmp

def gen_vcv(start='',end='',sh_path=DataPath.sh_min,sz_path=DataPath.sz_min):
    warnings.filterwarnings('ignore')
    tar_ticker,tar_date,tar_df=read_data(start,end)
    sz_tar_list=tar_df[tar_df.TICKER.str.endswith('.SZ')]['DATE'].drop_duplicates().to_list()
    sh_tar_list=tar_df[tar_df.TICKER.str.endswith('.SH')]['DATE'].drop_duplicates().to_list()
    res = []
    for path,tar_list in {sh_path:sh_tar_list, sz_path:sz_tar_list}.items():
        if tar_list==[]:
            continue
        tmp = Parallel(n_jobs=12)(delayed(cal_data)(file,path, tar_ticker) for file in tqdm(tar_list))
        res.extend(tmp)
    res = pd.concat(res).reset_index(drop=True)
    return res

def update(today='20250820'):
    save_path_old = DataPath.save_path_old
    save_path_update = DataPath.save_path_update
    flag = True
    while flag:
        if os.path.exists(os.path.join(save_path_update, 'vcv_daily.feather')):
            old_df = feather.read_dataframe(os.path.join(save_path_update, 'vcv_daily.feather'))
            old_date_list = old_df.DATE.drop_duplicates().to_list()
            old_date_list = sorted(old_date_list)
            new_start = old_date_list[old_date_list.index(old_df.DATE.max()) - 20]
            print('因子vcv更新中：')
            data = gen_vcv(start=new_start, end=today)  # todo: 每次更新检查，改end
            print('------------------------')
            test_df = old_df.merge(data, on=['DATE', 'TICKER'], how='inner')
            print(test_df)
            if np.isclose(test_df.iloc[:, 2], test_df.iloc[:, 3]).all():
                new_df = data[data.DATE > old_df.DATE.max()]
                full_df = pd.concat([old_df, new_df]).reset_index(drop=True)
                print(full_df)
                feather.write_dataframe(full_df, os.path.join(save_path_update, 'vcv_daily.feather'))
                feather.write_dataframe(full_df,os.path.join(DataPath.factor_out_path,'vcv_daily.feather'))
            else:
                print('检查更新，数据有问题！')
                exit()
            flag = False
        else:
            print('新因子vcv生成中：')
            data = gen_vcv(start='20200101', end='20221231')  # 第一次生成因子
            feather.write_dataframe(data, os.path.join(save_path_old, 'vcv_daily.feather'))
            feather.write_dataframe(data, os.path.join(save_path_update, 'vcv_daily.feather'))

if __name__=='__main__':
    update()