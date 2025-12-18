import warnings
import psutil
import os
import feather
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from tqdm import tqdm
import statsmodels.api as sm
from tool_tyx.path_data import DataPath
import numpy as np
import pandas as pd

def get_auc_data(date):
    df = feather.read_dataframe(fr'C:\Users\admin\Desktop\{date}.feather')
    df = df[(df['tradetime'] >= 92000000) & (df['tradetime'] < 92459950)]
    df['sec'] = df['tradetime'] // 1000
    df.sort_values(['TICKER', 'tradetime'], inplace=True)
    split_by_sec = df.groupby(['TICKER', 'sec']).agg(
        {'当前撮合成交量': ['first', 'last'], 'Price': ['first', 'last', 'max', 'min']})
    split_by_sec.columns = ['vol_first', 'vol_last', 'open', 'close', 'high', 'low']
    split_by_sec['volume_diff'] = split_by_sec['vol_last'] - split_by_sec['vol_first']
    split_by_sec.reset_index(inplace=True)
    split_by_sec['DATE'] = date
    split_by_sec.drop(columns=['vol_first', 'vol_last'], inplace=True)
    split_by_sec.rename(columns={'sec': 'time', 'volume_diff': 'volume'}, inplace=True)
    return split_by_sec


# 杀死多进程
def kill_mutil_process(pid_list=[]):
    if len(pid_list) == 0:
        current_pids = psutil.pids()
        # 找出使用Python的进程
        python_pids = [
            pid for pid in current_pids
            if 'python.exe' in psutil.Process(pid=pid).name()
        ]
        # python_cmd = [psutil.Process(pid=pid).cmdline() for pid in python_pids]  # 查看所有python进程
        python_multil_process = [
            pid for pid in python_pids if '--multiprocessing-fork' in str(
                psutil.Process(pid=pid).cmdline())
        ]
    else:
        python_multil_process = pid_list
    for pid in python_multil_process:
        try:
            # print('Killing process with pid {}'.format(pid))
            psutil.Process(pid=pid).terminate()  # 关闭进程
        except:
            continue

def get_tar_date(start,end):
    calender = pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'))
    calender['trade_date'] = calender['trade_date'].astype(str)
    calender = calender[(calender.trade_date >= start) & (calender.trade_date <= end)]
    tar_date = sorted(list(calender.trade_date.unique()))
    return tar_date

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

def update_muli(filename,today,run,num=-50):
    if os.path.exists(os.path.join(DataPath.save_path_update,filename)):
    # if False:
        print('因子更新中')
        old=feather.read_dataframe(os.path.join(DataPath.save_path_update,filename))
        new_start=sorted(old.DATE.drop_duplicates().to_list())[num]
        res=run(start=new_start,end=today)
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                # feather.write_dataframe(tmp, os.path.join(r'C:\Users\admin\Desktop', col + '.feather'))
                old=feather.read_dataframe(os.path.join(DataPath.save_path_update,col+'.feather'))
                test=old.merge(tmp,on=['DATE','TICKER'],how='inner').dropna()
                test.sort_values(['TICKER','DATE'],inplace=True)
                tar_list = sorted(list(test.DATE.unique()))[-5:]
                test = test[test.DATE.isin(tar_list)]
                if np.isclose(test.iloc[:,2],test.iloc[:,3]).all():
                    tmp=tmp[tmp.DATE>old.DATE.max()]
                    old=pd.concat([old,tmp]).reset_index(drop=True).drop_duplicates()
                    print(old)
                    feather.write_dataframe(old,os.path.join(DataPath.save_path_update,col+'.feather'))
                    # feather.write_dataframe(old,os.path.join(DataPath.factor_out_path,col+'.feather'))
                else:
                    print(test[~np.isclose(test.iloc[:,2],test.iloc[:,3])])
                    # tt=test[~np.isclose(test.iloc[:, 2], test.iloc[:, 3])]
                    # feather.write_dataframe(tt,r'C:\Users\admin\Desktop\tt.feather')
                    print('检查更新出错!')
                    exit()
    else:
        print('因子生成中')
        res=run(start='20200101',end='20221231')
        # res=run(start='20200101',end='20250822')
        # res=run(start='20250828',end='20250829')
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                print(tmp)
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))
                # feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))


def read_min_data(date):
    try:
        df_sz=feather.read_dataframe(os.path.join(DataPath.sz_min,date+'.feather'))
    except FileNotFoundError:
        df_sz=pd.DataFrame()
    try:
        df_sh=feather.read_dataframe(os.path.join(DataPath.sh_min,date+'.feather'))
    except FileNotFoundError:
        df_sh=pd.DataFrame()
    df=pd.concat([df_sh,df_sz]).reset_index(drop=True)
    return df


def read_ml_data(date,path):
    x=feather.read_dataframe(os.path.join(path,f'{date}_train_x.feather'))
    x.replace([np.inf,-np.inf],np.nan,inplace=True)
    y=feather.read_dataframe(os.path.join(path,f'{date}_train_y_ori.feather'))
    y.replace([np.inf,-np.inf],np.nan,inplace=True)
    return [x,y]

def get_ml_data(path):
    warnings.filterwarnings('ignore')
    tar_list = sorted(list(set([x[:8] for x in os.listdir(path)])))
    train_list=[x for x in tar_list if x>='20220104' and x<='20240201']
    valid_list=[x for x in tar_list if x>='20240202' and x<='20240731' and x not in ['20240206', '20240207', '20240208']]
    test_list=[x for x in tar_list if x not in train_list and x not in valid_list]
    train_data_list=Parallel(n_jobs=15)(delayed(read_ml_data)(date,path) for date in tqdm(train_list,desc='train datas are preparing'))
    train_data_list1=pd.concat([x[0] for x in train_data_list]).set_index(['TICKER','DATE'])
    train_data_list2=pd.concat([x[1] for x in train_data_list]).set_index(['TICKER','DATE'])
    train_data_list=[train_data_list1,train_data_list2]

    valid_data_list = Parallel(n_jobs=15)(delayed(read_ml_data)(date,path) for date in tqdm(valid_list,desc='valid datas are preparing'))
    valid_data_list1 = pd.concat([x[0] for x in valid_data_list]).set_index(['TICKER','DATE'])
    valid_data_list2 = pd.concat([x[1] for x in valid_data_list]).set_index(['TICKER','DATE'])
    valid_data_list=[valid_data_list1,valid_data_list2]

    test_data_list = Parallel(n_jobs=15)(delayed(read_ml_data)(date,path) for date in tqdm(test_list, desc='test datas are preparing'))
    test_data_list1 = pd.concat([x[0] for x in test_data_list]).set_index(['TICKER', 'DATE'])
    test_data_list2 = pd.concat([x[1] for x in test_data_list]).set_index(['TICKER', 'DATE'])
    test_data_list = [test_data_list1, test_data_list2]

    return train_data_list,valid_data_list,test_data_list

def neutralize_factor(df, factor_col, mkt_cap_col,label='neutral'):
    # df['log_mkt'] = np.log(df[mkt_cap_col])
    # region 改2
    neutralized = []
    for date, date_df in df.groupby('DATE'):  # 截面中性化
        date_df.dropna(inplace=True)
        if date_df.empty:
            continue
        # print(date_df)
        X = sm.add_constant(date_df[mkt_cap_col])
        y = date_df[factor_col]
        # weights = np.sqrt(date_df[mkt_cap_col])

        model = sm.WLS(y, X).fit()
        date_df[f'{factor_col}_{label}'] = model.resid
        neutralized.append(date_df)
    if len(neutralized) > 0:
        return pd.concat(neutralized)
    else:
        return pd.DataFrame()
    # endregion

def orth(data):
    ## step5:构建对称正交变换函数
    data_ = data.copy()  # 创建副本不影响原数据
    col = data_.columns.tolist()
    F = np.asmatrix(data_[col])  # 将数据框转化为矩阵
    M = F.T @ F  # 等价于 (F.shape[0] - 1) * np.cov(F.T)
    a, U = np.linalg.eig(M)  # a为特征值，U为特征向量
    D_inv = np.linalg.inv(np.diag(a))
    S = U @ np.sqrt(D_inv) @ U.T
    data_[col] = data_[col].dot(S)
    # data_[[x+'adjust' for x in col]] = data_[col].dot(S)
    return data_

# def orth(data):
#     pca=PCA()
#     data_=pca.fit_transform(data)
#     return pd.DataFrame(data_,columns=data.columns)

def change_file(file_name,col_name):
    tt = feather.read_dataframe(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file_name}.feather')
    col=np.setdiff1d(tt.columns,['DATE','TICKER'])[0]
    tt.rename(columns={col: col_name}, inplace=True)
    feather.write_dataframe(tt, fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file_name}.feather')

    tt = feather.read_dataframe(fr'D:\tyx\检查更新\2021.6-2025.8\{file_name}.feather')
    tt.rename(columns={col: col_name}, inplace=True)
    feather.write_dataframe(tt, fr'D:\tyx\检查更新\2021.6-2025.8\{file_name}.feather')