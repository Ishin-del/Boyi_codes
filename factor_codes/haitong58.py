# @Author: Yixin Tian
# @File: haitong58.py
# @Date: 2025/9/16 9:22
# @Software: PyCharm
import os
import warnings
import datetime
import statsmodels.api as sm
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import feather
from tqdm import tqdm

from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import read_min_data, get_tar_date,process_na_stock,update_muli


# 改1
def read_data(date):
    weekday=datetime.date(int(date[:4]), int(date[4:6]), int(date[6:])).isoweekday()
    if weekday==5:
        return
    warnings.filterwarnings('ignore')
    df=read_min_data(date)
    if df.empty:
        return
    df=df[['TICKER','min','close','amount','active_sell_amount','active_buy_amount']]
    df=df[(df['min']<1457)&(df['min']!=930)]
    df.sort_values(['TICKER','min'],inplace=True)
    df['min_ret']=df.groupby('TICKER')['close'].pct_change()
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df['pre_min_ret']=df.groupby('TICKER')['min_ret'].shift(1)
    df['weekday']=weekday
    df['time_periods']=np.where(df['min']<=959,1,np.where(df['min']>=1427,3,2))
    df['DATE']=date
    return df

def mulit_reg(df):
    x_col = ['pre_min_ret', 'weekday', 'time_periods']
    y_col = 'min_ret'
    df.dropna(subset=['min_ret', 'pre_min_ret', 'weekday','time_periods'],how='any',axis=0,inplace=True)
    X = sm.add_constant(df[x_col])
    y = df[y_col]
    model = sm.WLS(y, X).fit()
    df['resid'] = model.resid
    return df

def cal(tmp_date,tmp_df):
    day=tmp_date[-1]
    # print(len(tmp_date))
    tmp_df.sort_values(['TICKER','DATE','min'],inplace=True)
    tmp_df=tmp_df.groupby('TICKER').apply(mulit_reg).reset_index(drop=True)
    tmp_df['flag']=np.where(tmp_df['resid']>0,'know_act_sell','know_act_buy') #todo:??
    tmp1=tmp_df[tmp_df.DATE==day].groupby('TICKER')['amount'].sum().reset_index().rename(columns={'amount':'day_amount'})
    tmp2=tmp_df[tmp_df.DATE==day].groupby(['TICKER','time_periods'])['amount'].sum().reset_index().pivot(index='TICKER',columns='time_periods',values='amount').reset_index()
    tmp2.columns=['TICKER','after_open','middle','before_close']
    tmp5=tmp_df[tmp_df.DATE==day].groupby(['TICKER','time_periods']).agg({'active_sell_amount':'sum','active_buy_amount':'sum'}).reset_index().pivot(index='TICKER',columns=['time_periods'],values=['active_sell_amount','active_buy_amount']).reset_index()
    tmp5.columns=['TICKER','day_buy_open','day_buy_mid','day_buy_close','day_sell_open','day_sell_mid','day_sell_close']
    tmp3=tmp_df.groupby(['TICKER','flag'])['amount'].sum().reset_index().pivot(index=['TICKER'],columns='flag',values='amount').reset_index()
    tmp3.columns=['TICKER','know_act_buy_amount','know_act_sell_amount']
    tmp4=tmp_df.groupby(['TICKER','flag','time_periods'])['amount'].sum().reset_index().pivot(index='TICKER',columns=['flag','time_periods'],values='amount').reset_index()
    tmp4.columns=['TICKER','actbuy_open','actbuy_mid','actbuy_close','actsell_open','actsell_mid','actsell_close']
    tmp=tmp1.merge(tmp2,on='TICKER',how='inner').merge(tmp3,on='TICKER',how='inner').merge(tmp4,on='TICKER',how='inner').merge(tmp5,on='TICKER',how='inner')
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
    tmp=tmp[['DATE','TICKER','sell_trade_ratio','sell_open_ratio','sell_midd_ratio','sell_close_ratio',
                    'sell_day_open_ratio','sell_day_midd_ratio','sell_day_close_ratio','buy_trade_ratio',
                    'buy_open_ratio','buy_midd_ratio','buy_close_ratio','buy_day_open_ratio','buy_day_midd_ratio',
                    'buy_day_close_ratio','net_trade_ratio','net_open_ratio','net_midd_ratio','net_close_ratio',
                    'net_day_open_ratio','net_day_midd_ratio','net_day_close_ratio']]
    feather.write_dataframe(tmp,os.path.join(DataPath.tmp_path,'haitong58',f'{day}.feather'))

def run(start,end):
    warnings.filterwarnings('ignore')
    tar_list=get_tar_date(start,end)
    basic_file = r'C:\Users\admin\Desktop\haitong58'
    os.makedirs(basic_file,exist_ok=True)
    min_date_list = []



    pp = r'C:\Users\admin\Desktop\tt.feather'
    if os.path.exists(pp):
        df = feather.read_dataframe(r'C:\Users\admin\Desktop\tt.feather')
        if start>df.DATE.max():
            old_df= feather.read_dataframe(r'C:\Users\admin\Desktop\tt.feather')
            df=Parallel(n_jobs=10)(delayed(read_data)(date) for date in tqdm(tar_list))
            df = pd.concat(df).reset_index(drop=True)
            df=pd.concat([old_df,df]).reset_index(drop=True)
            feather.write_dataframe(df, r'C:\Users\admin\Desktop\tt.feather')
    else:
        df = Parallel(n_jobs=10)(delayed(read_data)(date) for date in tqdm(tar_list))
        df = pd.concat(df).reset_index(drop=True)
        feather.write_dataframe(df, r'C:\Users\admin\Desktop\tt.feather')
        print('skjdhfkjf')

    df = process_na_stock(df, col='pre_min_ret')
    df.sort_values(['TICKER','DATE','min'],inplace=True)
    tar_date=sorted(df.DATE.unique().tolist())
    # print(tar_date)
    # res=[]
    print('cal factor:')
    for i in tqdm(range(len(tar_date)-19)):
        tmp_date = tar_date[i:i + 20]
        tmp_df = df[df.DATE.isin(tmp_date)]
        cal(tmp_date,tmp_df)
    # res=pd.concat(res).reset_index(drop=True)
    # print('cal...')
    # Parallel(n_jobs=5)(delayed(cal)(tar_date[i:i + 20],df[df.DATE.isin(tar_date[i:i + 20])])for i in tqdm(range(len(tar_date)-19)))
    # res = pd.concat(res).reset_index(drop=True)
    # return res

# def run(start,end):
#     df=cal(start,end)
#     df=pd.concat(df).reset_index(drop=True)
#     df = process_na_stock(df, col='sell_trade_ratio')
#     res = []
#     for col in df.columns.tolist()[2:]:
#         tmp = df[['DATE', 'TICKER', col]]
#         tmp[col + '_roll20'] = tmp.groupby('TICKER')[col].rolling(20, 5).mean().values
#         tmp = tmp[['DATE', 'TICKER', col + '_roll20']]
#         res.append(tmp)
#     return res

def update(today='20250905'):
    update_muli('sell_trade_ratio_roll20.feather',today,run)

if __name__=='__main__':
    # update()
    run('20200101', '20211231')