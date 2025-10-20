import os
import feather
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
from tool_tyx.path_data import DataPath
from datetime import datetime

def process_na_value(df):
    '''解决某些股票出现时间序列断开的情况，这种情况会在rolling（window!=min_period）的时候,因子值出现不同'''
    pt = df.pivot_table(columns='TICKER', index='DATE', values='stock_ret')
    pt = pt.ffill()
    pp = pt.melt(ignore_index=False).reset_index(drop=False)
    df = pd.merge(pp, df, how='left', on=['TICKER', 'DATE'])
    use_cols = list(np.setdiff1d(df.columns, ['TICKER', 'DATE']))
    for c in use_cols:
        df[c] = df.groupby('TICKER')[c].ffill()
    return df


def helper(file, path, tar_date, tar_ticker):
    if file[:8] in list(tar_date):
        tmp = feather.read_dataframe(os.path.join(path, file))[['TICKER', 'min', 'close']]
        tmp = tmp[tmp.TICKER.isin(tar_ticker)]
        # 因子计算：
        tmp = tmp[(tmp['min'] > 930) & (tmp['min'] < 1457)]
        tmp.sort_values(['TICKER', 'min'], inplace=True)
        tmp['min_ret'] = tmp.groupby('TICKER')['close'].pct_change()
        tmp = tmp[(tmp != np.inf).all(axis=1)]
        tmp = tmp[(tmp != -np.inf).all(axis=1)]
        tmp = tmp.groupby('TICKER')['min_ret'].agg('std').reset_index()
        tmp.rename(columns={'min_ret': 'volatility'}, inplace=True)
        tmp['DATE'] = file[:8]
        return tmp
    else:
        return


def get_min_ret(tar_date, tar_ticker, sh_tar_list, sz_tar_list,
                sh_path=DataPath.sh_min, sz_path=DataPath.sz_min, data_path=DataPath.save_help_path):
    warnings.filterwarnings(action='ignore')
    # 根据分钟频 拉取日频分钟收益率的波动率
    res = []
    print('计算股票分钟收益率波动：')
    for path, tar_list in {sh_path: sh_tar_list, sz_path: sz_tar_list}.items():
        if tar_list == []:
            continue
        tmp = Parallel(n_jobs=15)(
            delayed(helper)(file + '.feather', path, tar_date, tar_ticker) for file in tqdm(tar_list))
        res.extend(tmp)
    res = pd.concat(res).reset_index(drop=True)
    return res


def get_indivi_ratio(date, path, tar_ticker):
    try:
        tmp = feather.read_dataframe(os.path.join(path, date + '.feather'))
    except FileNotFoundError:
        return
    tmp = tmp[tmp.TICKER.isin(tar_ticker)]
    tmp['small_amount'] = (tmp['small_sell_amount'] + tmp['small_buy_amount']) / 2
    tmp['res_amount'] = (tmp['medium_sell_amount'] + tmp['large_sell_amount'] + tmp['xlarge_sell_amount'] +
                         tmp['medium_buy_amount'] + tmp['large_buy_amount'] + tmp['xlarge_buy_amount']) / 2
    tmp['indivi_ratio'] = tmp['small_amount'] / (tmp['small_amount'] + tmp['res_amount'])
    return tmp[['TICKER', 'DATE', 'indivi_ratio']]


def read_data(start='', end='',
              df_path=DataPath.daily_path,
              mkt_path=DataPath.mkt_index,
              moneyflow_sh=DataPath.moneyflow_sh,
              moneyflow_sz=DataPath.moneyflow_sz
              ):
    df = feather.read_dataframe(df_path, columns=['TICKER', 'DATE', 'close'])
    # 仅作测试用
    # df=df[df.TICKER=='000001.SZ']
    # todo:
    # df=df[df.TICKER=='600000.SH']
    df.sort_values(['TICKER', 'DATE'], inplace=True)
    df = df.groupby('TICKER').filter(lambda x: len(x) > 20)
    df = df[(df['DATE'] <= end) & (df['DATE'] >= start)]
    df.sort_values(['TICKER', 'DATE'], inplace=True)
    df['stock_ret'] = df.groupby('TICKER')['close'].pct_change()
    df.drop(columns='close', inplace=True)

    tar_ticker = df.TICKER.drop_duplicates().to_list()
    tar_date = df.DATE.drop_duplicates().to_list()
    tar_date = sorted(tar_date)
    tar_df = df[['TICKER', 'DATE']].drop_duplicates()
    sz_tar_list = tar_df[tar_df.TICKER.str.endswith('.SZ')]['DATE'].drop_duplicates().to_list()
    sh_tar_list = tar_df[tar_df.TICKER.str.endswith('.SH')]['DATE'].drop_duplicates().to_list()

    # 读取 & 计算市场数据
    mkt_sh = pd.read_csv(os.path.join(mkt_path, '000001.SH.csv'))[['time', 'close', 'preClose']]
    mkt_sh['DATE'] = mkt_sh['time'].str.replace('-', '')
    # mkt_sh['DATE'] = mkt_sh['time'].apply(lambda x: datetime.strptime(x, "%Y/%m/%d").strftime("%Y%m%d"))
    mkt_sh['mkt_ret'] = mkt_sh['close'] / mkt_sh['preClose'] - 1
    mkt_sh = mkt_sh[['DATE', 'mkt_ret']]

    mkt_sz = pd.read_csv(os.path.join(mkt_path, '399001.SZ.csv'))[['time', 'close', 'preClose']]
    mkt_sz['DATE'] = mkt_sz['time'].str.replace('-', '')
    mkt_sz['mkt_ret'] = mkt_sz['close'] / mkt_sz['preClose'] - 1
    mkt_sz = mkt_sz[['DATE', 'mkt_ret']]

    mkt_sz = df[df.TICKER.str.endswith('.SZ')].merge(mkt_sz, on='DATE', how='left')
    mkt_sh = df[df.TICKER.str.endswith('.SH')].merge(mkt_sh, on='DATE', how='left')
    df = pd.concat([mkt_sz, mkt_sh]).reset_index(drop=True)
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    # 计算惊恐度
    df['fear_degree'] = (abs((df['stock_ret'] - df['mkt_ret'])) / (abs(df['stock_ret'])) + abs(df['mkt_ret']) + 0.1)
    # df = df[(df != np.inf).all(axis=1)]

    # 计算股票ret的std
    tmp = get_min_ret(tar_date, tar_ticker, sh_tar_list, sz_tar_list)
    df = df.merge(tmp, on=['DATE', 'TICKER'], how='left')
    # df.dropna(how='any', axis=0, inplace=True)
    # 拉取个人投资者比例
    tmp = []
    print('拉取&计算保存个人投资者比例:')
    for path, tar_list in {moneyflow_sh: sh_tar_list, moneyflow_sz: sz_tar_list}.items():
        if tar_list == []:
            continue
        t = Parallel(n_jobs=15)(delayed(get_indivi_ratio)(date, path, tar_ticker) for date in tqdm(tar_list))
        tmp.extend(t)
    tmp = pd.concat(tmp).reset_index(drop=True)
    df = df.merge(tmp, on=['TICKER', 'DATE'], how='left')
    df = process_na_value(df)
    return df

def get_rank(x,rank):
    x = x.sort_values(by=['DATE']).reset_index(drop=True)
    return x.iloc[rank:,:]

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def gen_fear_factor(df):
    df.sort_values(['TICKER', 'DATE'], inplace=True)
    # 注意力衰减: 只筛选出来惊恐度大于前两天惊恐度均值作为权重，删除小于的数据
    df = df.reset_index(drop=True)
    df['prev_mean'] = df.groupby('TICKER')['fear_degree'].transform(lambda x: x.shift(1).rolling(window=2).mean())
    df['fear_diff'] = df['fear_degree'] - df['prev_mean']

    # df['fear_adj'] = df['fear_diff'].where(df['fear_diff'] > 0)
    df['fear_adj']=sigmoid(df['fear_diff'])
    df.drop(columns=['prev_mean', 'fear_diff'], inplace=True)
    # df.dropna(axis=0, inplace=True, how='any')
    # 草木皆兵因子
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df['tree_soldier'] = df['fear_adj'] * df['volatility'] * df['indivi_ratio']
    df.sort_values(['TICKER', 'DATE'], inplace=True)
    df=process_na_value(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['tree_soldier_ret'] = df.groupby('TICKER')['tree_soldier'].apply(
        lambda x: x.rolling(window=20, min_periods=5).mean()).values
    df['tree_soldier_vol'] = df.groupby('TICKER')['tree_soldier'].apply(
        lambda x: x.rolling(window=20, min_periods=5).std()).values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['fear_idvi_fac'] = (df['tree_soldier_ret'] + df['tree_soldier_vol']) / 2
    df = df.groupby('TICKER').apply(get_rank,50).reset_index(drop=True)
    # print(np.isinf(df).any().any())
    df1 = df[['TICKER', 'DATE', 'tree_soldier_ret']]
    df2 = df[['TICKER', 'DATE', 'tree_soldier_vol']]
    df3 = df[['TICKER', 'DATE', 'fear_idvi_fac']]
    # feather.write_dataframe(df,os.path.join(r'C:\Users\admin\Desktop\test.feather'))
    return df1, df2, df3

def update(today='20250904'):
    save_path_old = DataPath.save_path_old
    save_path_update = DataPath.save_path_update
    flag = True
    while flag:
        if os.path.exists(os.path.join(save_path_update, 'tree_soldier_ret.feather')):
            old_df = feather.read_dataframe(os.path.join(save_path_update, 'tree_soldier_ret.feather'))
            new_start = sorted(old_df.DATE.drop_duplicates().to_list())[-70]
            # new_start='20221001'
            print('因子fear更新中:')
            df = read_data(start=new_start, end=today)  # todo: 每次更新检查，改end
            df1, df2, df3 = gen_fear_factor(df)
            for name, data in {'tree_soldier_ret': df1, 'tree_soldier_vol': df2, 'fear_idvi_fac': df3}.items():
                old_df = feather.read_dataframe(os.path.join(save_path_update, name + '.feather'))
                # test_date = old_date_list[old_date_list.index(old_df.DATE.max()) - 15:]
                test_df = old_df.merge(data, on=['DATE', 'TICKER'], how='inner').dropna()
                test_df.sort_values(['TICKER', 'DATE'], inplace=True)
                tar_list = sorted(list(test_df.DATE.unique()))[-5:]
                test_df = test_df[test_df.DATE.isin(tar_list)]
                if np.isclose(test_df.iloc[:, 2], test_df.iloc[:, 3]).all():
                    new_df = data[data.DATE > old_df.DATE.max()]
                    full_df = pd.concat([old_df, new_df]).reset_index(drop=True)
                    print(full_df)
                    feather.write_dataframe(full_df, os.path.join(save_path_update, name + '.feather'))
                    feather.write_dataframe(full_df,os.path.join(DataPath.factor_out_path,name + '.feather'))
                else:
                    print(test_df[~np.isclose(test_df.iloc[:,2],test_df.iloc[:,3])])
                    print('检查更新，数据有问题！')
                    exit()

        else:
            print('新相关因子fear生成中:')
            df = read_data(start='20200101', end='20221231')  # '20221231'
            df1, df2, df3 = gen_fear_factor(df)  # 第一次生成因子
            for name, data in {'tree_soldier_ret': df1, 'tree_soldier_vol': df2, 'fear_idvi_fac': df3}.items():
                print(data)
                feather.write_dataframe(data, os.path.join(save_path_old, name + '.feather'))
                feather.write_dataframe(data, os.path.join(save_path_update, name + '.feather'))
                # break
        flag = False

if __name__ == '__main__':
    update('20251016')
    # update()
    # df = read_data(start='20200101', end='20250822')
    # df = read_data(start='20200101', end='20211231')
    # df1,df2,df3=gen_fear_factor(df)
    # count=1
    # for df in [df1,df2,df3]:
    #     feather.write_dataframe(df,fr'C:\Users\admin\Desktop\test{count}.feather')
    #     count+=1
    # ---------------------------
    # old_df = feather.read_dataframe(os.path.join(r'C:\Users\admin\Desktop', 'test1.feather'))
    # old_date_list = old_df.DATE.drop_duplicates().to_list()
    # old_date_list = sorted(old_date_list)
    # # new_start = old_date_list[max(old_date_list.index(len(old_date_list)-1) - 80, 0)]
    # new_start = old_date_list[-50]
    # df = read_data(start=new_start, end='20220601')
    # df1,df2,df3=gen_fear_factor(df)
    # count=1
    # for df in [df1,df2,df3]:
    #     feather.write_dataframe(df,fr'C:\Users\admin\Desktop\new{count}.feather')
    #     count+=1