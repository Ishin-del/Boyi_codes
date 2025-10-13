# @Author: Yixin Tian
# @File: haitong58.py
# @Date: 2025/9/16 9:22
# @Software: PyCharm
import os
import time
import warnings
import datetime
import statsmodels.api as sm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import feather
from tqdm import tqdm
import gc
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import read_min_data,process_na_stock


# 改1
def read_data(date):
    warnings.filterwarnings('ignore')
    weekday=datetime.date(int(date[:4]), int(date[4:6]), int(date[6:])).isoweekday()
    df=read_min_data(date)
    if df.empty:
        return
    df=df[['TICKER','min','close','amount','active_sell_amount','active_buy_amount']]
    df=df[(df['min']<1457)&(df['min']!=930)]
    df.sort_values(['TICKER','min'],inplace=True)
    df = df.reset_index(drop=True)
    df['min_ret']=df.groupby('TICKER')['close'].pct_change()
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df['pre_min_ret']=df.groupby('TICKER')['min_ret'].shift(1)
    df['weekday']=weekday
    df['time_periods']=np.where(df['min']<=959,1,np.where(df['min']>=1427,3,2))
    df['DATE']=date
    return df

def calc_residuals(df, x_col, y_col):
    X_all = df[x_col].to_numpy(dtype=np.float32)
    y_all = df[y_col].to_numpy(dtype=np.float32)
    tickers = df['TICKER'].to_numpy()

    # 结果数组
    resid_all = np.empty_like(y_all)

    uniq_ticker, idx_start = np.unique(tickers, return_index=True)
    idx_order = np.argsort(idx_start)
    uniq_ticker = uniq_ticker[idx_order]
    idx_start = idx_start[idx_order]
    idx_end = np.r_[idx_start[1:], len(tickers)]

    for s, e in zip(idx_start, idx_end):
        X = X_all[s:e]
        y = y_all[s:e]
        if len(y) == 0:
            resid_all[s:e] = np.nan
            continue
        beta = np.linalg.pinv(X) @ y
        y_hat = X @ beta
        resid_all[s:e] = y - y_hat

    df = df.copy()
    df['resid'] = resid_all
    return df


def mulit_reg(df,range_vars,week_var):
    # df.sort_values(['TICKER','DATE','min'],inplace=True)
    df = df.reset_index(drop=True)
    # week_var = df['weekday'].unique()
    # week_var = week_var.tolist()
    # week_var.sort()
    # df[week_var] = pd.get_dummies(df['weekday'])[week_var]
    # df[week_var] =df[week_var].astype(int)
    #
    # range_var = df['time_periods'].unique()
    # range_var = range_var.tolist()
    # range_var.sort()
    # range_vars = [str(x) + 'rg' for x in range_var]
    # df[range_vars] = pd.get_dummies(df['time_periods'])[range_var]
    # df[range_vars] =df[range_vars].astype(int)

    x_col = ['pre_min_ret'] + range_vars + week_var
    y_col = 'min_ret'
    # df.dropna(subset=x_col + [y_col],how='any',axis=0,inplace=True)
    # X = sm.add_constant(df[x_col])
    # y = df[y_col]
    # if len(X)==0 or len(y)==0:
    #     return
    # model = sm.WLS(y, X).fit()
    # df['resid'] = model.resid

    X, y = df[x_col].values, df[y_col].values
    if len(X) == 0 or len(y) == 0:
        return
    X, y = np.float32(X), np.float32(y)
    df['resid'] = y - X.dot(np.linalg.pinv(X).dot(y))

    # df = df.drop(range_vars,axis=1)
    # df = df.drop(week_var,axis=1)
    return df


def cal(tmp_df):
    day=tmp_df[-1]['DATE'].values[0]
    print(day)
    # try:
    #     day=tmp_df[-1]['DATE'].iloc[0]
    # except:
    #     print(tmp_df)
    if os.path.exists(os.path.join(DataPath.tmp_path,'haitong58',f'{day}.feather')):
        print(f'{day}文件已存在！')
        return
    # print(len(tmp_date))
    tmp_df = pd.concat(tmp_df)
    # tmp_df.sort_values(['TICKER','DATE','min'],inplace=True)
    # tmp_df['weekday'].unique()
    # tqdm.pandas()

    # 没看具体逻辑
    tmp_df.dropna(axis=0,how='any',inplace=True)
    # todo:
    week_var = tmp_df['weekday'].unique()
    week_var = week_var.tolist()
    week_var.sort()
    tmp_df[week_var] = pd.get_dummies(tmp_df['weekday'])[week_var]
    tmp_df[week_var] = tmp_df[week_var].astype(int)

    range_var = tmp_df['time_periods'].unique()
    range_var = range_var.tolist()
    range_var.sort()
    range_vars = [str(x) + 'rg' for x in range_var]
    tmp_df[range_vars] = pd.get_dummies(tmp_df['time_periods'])[range_var]
    tmp_df[range_vars] = tmp_df[range_vars].astype(int)

    # t1=time.time()
    tmp_df.sort_values(['TICKER', 'DATE', 'min'], inplace=True)
    tmp_df = tmp_df.reset_index(drop=True)
    # tmp_df=tmp_df.groupby('TICKER').apply(mulit_reg,range_vars,week_var).reset_index(drop=True)
    tmp_df = calc_residuals(tmp_df,x_col=['pre_min_ret'] + range_vars + week_var,
                                       y_col='min_ret')
    # todo:
    tmp_df.drop(columns=range_vars+week_var,inplace=True)
    # print(tmp_df.columns)
    # t2=time.time()
    # print(f'回归时间：{t2-t1}s')

    tmp_df['flag']=np.where(tmp_df['resid']>0,'know_act_sell','know_act_buy') #todo:??
    help=tmp_df[tmp_df.DATE==day]

    tmp1=help.groupby('TICKER')['amount'].sum().reset_index().rename(columns={'amount':'day_amount'})
    tmp2 = (help.groupby(['TICKER', 'time_periods']).agg({'active_sell_amount': 'sum', 'active_buy_amount': 'sum', 'amount': 'sum'}).reset_index()
            .pivot(index='TICKER', columns='time_periods',values=['active_sell_amount', 'active_buy_amount', 'amount']).reset_index()).set_index('TICKER')
    tmp2.columns = ['day_sell_open', 'day_sell_mid', 'day_sell_close', 'day_buy_open', 'day_buy_mid',
                    'day_buy_close', 'after_open', 'middle', 'before_close']
    tmp3=tmp_df.groupby(['TICKER','flag']).agg({'amount':'sum'}).reset_index().pivot(index='TICKER',columns='flag',values='amount').reset_index().set_index('TICKER')
    tmp3.columns=['know_act_buy_amount','know_act_sell_amount']
    tmp4=tmp_df.groupby(['TICKER','flag','time_periods']).agg({'amount':'sum'}).reset_index().pivot(index='TICKER',columns=['flag','time_periods'],values='amount').reset_index().set_index('TICKER')
    tmp4.columns=['actbuy_open','actbuy_mid','actbuy_close','actsell_open','actsell_mid','actsell_close']
    tmp=tmp1.merge(tmp2,on='TICKER',how='inner').merge(tmp3,on='TICKER',how='inner').merge(tmp4,on='TICKER',how='inner').reset_index()
    # 因子计算：
    tmp['sell_trade_ratio']=tmp['know_act_sell_amount']/tmp['day_amount']
    tmp['sell_open_ratio']=tmp['actsell_open']/tmp['after_open']
    tmp['sell_midd_ratio']=tmp['actsell_mid']/tmp['middle']
    tmp['sell_close_ratio']=tmp['actsell_close']/tmp['before_close']
    tmp['sell_day_open_ratio'] = tmp['actsell_open'] / tmp['day_sell_open']
    tmp['sell_day_midd_ratio'] = tmp['actsell_mid'] / tmp['day_sell_mid']
    tmp['sell_day_close_ratio'] = tmp['actsell_close'] / tmp['day_sell_close']

    tmp['buy_trade_ratio']=tmp['know_act_buy_amount']/tmp['day_amount']

    tmp['buy_open_ratio'] = tmp['actbuy_open'] / tmp['after_open']
    tmp['buy_midd_ratio'] = tmp['actbuy_mid'] / tmp['middle']
    tmp['buy_close_ratio']= tmp['actbuy_close'] / tmp['before_close']

    tmp['buy_day_open_ratio'] = tmp['actbuy_open'] / tmp['day_buy_open']
    tmp['buy_day_midd_ratio'] = tmp['actbuy_mid'] / tmp['day_buy_mid']
    tmp['buy_day_close_ratio']= tmp['actbuy_close'] /tmp['day_buy_close']
    tmp['net_trade_ratio']=(tmp['know_act_buy_amount']-tmp['know_act_sell_amount'])/tmp['day_amount']
    tmp['net_open_ratio'] = (tmp['actbuy_open']-tmp['actsell_open']) / tmp['after_open']
    tmp['net_midd_ratio'] = (tmp['actbuy_mid']-tmp['actsell_mid']) / tmp['middle']
    tmp['net_close_ratio'] = (tmp['actbuy_close']-tmp['actsell_close']) / tmp['before_close']
    tmp['net_day_open_ratio'] = (tmp['actbuy_open'] - tmp['actsell_open']) / (tmp['day_buy_open']-tmp['day_sell_open'])
    tmp['net_day_midd_ratio'] = (tmp['actbuy_mid'] - tmp['actsell_mid']) / (tmp['day_buy_mid']-tmp['day_sell_mid'])
    tmp['net_day_close_ratio'] = (tmp['actbuy_close'] - tmp['actsell_close']) / (tmp['day_buy_close']-tmp['day_sell_close'])
    tmp['DATE']=day
    # tmp=tmp[['DATE','TICKER','sell_trade_ratio','sell_open_ratio','sell_midd_ratio','sell_close_ratio',
    #                 'sell_day_open_ratio','sell_day_midd_ratio','sell_day_close_ratio','buy_trade_ratio',
    #                 'buy_open_ratio','buy_midd_ratio','buy_close_ratio','buy_day_open_ratio','buy_day_midd_ratio',
    #                 'buy_day_close_ratio','net_trade_ratio','net_open_ratio','net_midd_ratio','net_close_ratio',
    #                 'net_day_open_ratio','net_day_midd_ratio','net_day_close_ratio']]
    tmp=tmp[['DATE','TICKER','sell_trade_ratio','sell_midd_ratio','sell_day_midd_ratio','sell_day_close_ratio',
                    'net_trade_ratio','net_midd_ratio']]
    # print(tmp)
    feather.write_dataframe(tmp,os.path.join(DataPath.tmp_path,'haitong58',f'{day}.feather'))

def cal2(tar_list):
    warnings.filterwarnings('ignore')
    already = os.listdir(os.path.join(DataPath.tmp_path,'haitong58'))
    already = [i[:8] for i in already]
    already.sort()
    already = already[:-19]
    tar_list = [i for i in tar_list if i >= max(already)]
    if tar_list==[]:
        return
    # basic_file = r'C:\Users\admin\Desktop\haitong58'
    # os.makedirs(basic_file,exist_ok=True)
    min_date_list = []

    for date in tqdm(tar_list):
        if len(min_date_list) < 20:
            df=read_data(date)
            if df is not None:
                min_date_list.append(df)
            # min_date_list=[x for x in min_date_list if x is not None]
        else:# >=20的时候
            # t1=time.time()
            # print('---------------------------')
            # print(min_date_list[-1])
            cal(min_date_list)
            df = read_data(date)
            if df is not None:
                min_date_list.append(df)
            del min_date_list[0]
            gc.collect()
            # t2=time.time()
            # print(f'总运行时间：{t2-t1}s')

def tmp_run(date):
    if os.path.exists(os.path.join(DataPath.tmp_path,'haitong58',f'{date}.feather')):
        tmp_df=feather.read_dataframe(os.path.join(DataPath.tmp_path,'haitong58',f'{date}.feather'))
        return tmp_df
    else:
        # cal2(date)
        print(f'{date}无数据')
        return

def get_tar_date(start,end):
    calender = pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'))
    calender['trade_date'] = calender['trade_date'].astype(str)
    new_end_index=calender[calender.trade_date == end].index + 1
    try:
        new_end=calender.iloc[new_end_index].trade_date.iloc[0]
    except:
        new_end=end
    # print(new_end)
    calender = calender[(calender.trade_date >= start) & (calender.trade_date <= new_end)]
    tar_date = sorted(list(calender.trade_date.unique()))
    return tar_date

def run(start, end):
    tar_list = get_tar_date(start, end)
    # print(tar_list)
    cal2(tar_list)
    df=Parallel(n_jobs=15)(delayed(tmp_run)(date) for date in tqdm(tar_list,desc='数据合并中'))
    df=pd.concat(df).reset_index(drop=True)
    df=df[['DATE','TICKER','sell_trade_ratio','sell_midd_ratio','sell_day_midd_ratio','sell_day_close_ratio',
                    'net_trade_ratio','net_midd_ratio']]
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    print(df)
    return [df]

def update(today='20250905'):
    update_muli('sell_trade_ratio.feather',today,run)

def update_muli(filename,today,run,num=-50):
    if os.path.exists(os.path.join(DataPath.save_path_update,filename)):
    # if False:
        print('因子更新中')
        # old=feather.read_dataframe(os.path.join(DataPath.save_path_update,filename))
        # new_start=sorted(old.DATE.drop_duplicates().to_list())[num]
        new_start='20220910'
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
                    old.replace([np.inf, -np.inf], np.nan, inplace=True)
                    old.dropna(inplace=True)
                    print(old)
                    feather.write_dataframe(old,os.path.join(DataPath.save_path_update,col+'.feather'))
                    feather.write_dataframe(old,os.path.join(DataPath.factor_out_path,col+'.feather'))
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
                tmp.replace([np.inf,-np.inf],np.nan,inplace=True)
                tmp.dropna(inplace=True)
                print(tmp)
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))
                # feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))

if __name__=='__main__':
    update('20250919')
    # tmp_run('20250919')
    # run('20250601','20250919')