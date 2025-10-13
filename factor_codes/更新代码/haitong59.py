# @Author: Yixin Tian
# @File: haitong59.py
# @Date: 2025/9/17 13:19
# @Software: PyCharm
import feather
import os
import pandas as pd
from joblib import Parallel, delayed
import warnings
from tqdm import tqdm
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import read_min_data, get_tar_date
import numpy as np

def read_data(date):
    warnings.filterwarnings('ignore')
    df=read_min_data(date)
    if df.empty:
        return
    df=df[['TICKER','active_buy_amount','active_sell_amount','amount','min']]
    help1=df[(df['min']>=1000)&(df['min']<=1426)] #盘中
    tmp1=help1.groupby('TICKER')['active_buy_amount'].sum().reset_index().rename(columns={'active_buy_amount':'act_buy_mid'})

    help2=df[(df['min']>=930)&(df['min']<959)] #开盘后
    help2['net_buy_open']=df['active_buy_amount']-df['active_sell_amount']
    tmp2=help2.groupby('TICKER').agg({'net_buy_open':['mean','std'],'amount':'sum'}).reset_index()
    tmp2.columns=['TICKER','net_buy_open_mean','net_buy_open_std','amount_open']
    tmp=tmp1.merge(tmp2, on='TICKER', how='inner')

    help3=df[(df['min']>=1427)&(df['min']<=1456)] #收盘前
    tmp3=help3.groupby('TICKER')['amount'].sum().reset_index().rename(columns={'amount':'amount_close'})
    tmp=tmp.merge(tmp3,on='TICKER',how='inner')
    tmp['DATE']=date
    return tmp[['DATE','TICKER','act_buy_mid','net_buy_open_mean','net_buy_open_std','amount_open','amount_close']]

def run(start,end):
    tar_list=get_tar_date(start,end)
    moneyflow = feather.read_dataframe(os.path.join(DataPath.to_df_path, 'moneyflow.feather'), columns=['DATE', 'TICKER',
                'large_sell_amount','medium_sell_amount','small_sell_amount','xlarge_sell_amount','large_buy_amount',
                'medium_buy_amount','small_buy_amount','xlarge_buy_amount'])
    moneyflow['amount'] = moneyflow.iloc[:, 2:].sum(axis=1)
    # 因子计算1:
    moneyflow['big_buy_ratio'] = moneyflow['large_buy_amount'] / moneyflow['amount']
    moneyflow['buy_concentration'] = (moneyflow['large_buy_amount'] ** 2 + moneyflow['medium_buy_amount'] ** 2 +
                moneyflow['small_buy_amount'] ** 2 + moneyflow['xlarge_buy_amount'] ** 2) /moneyflow['amount'] ** 2
    #
    df=Parallel(n_jobs=10)(delayed(read_data)(date) for date in tqdm(tar_list))
    df=pd.concat(df).reset_index(drop=True)
    df=df.merge(moneyflow,on=['DATE','TICKER'],how='left')
    # 因子计算2:
    # df['act_buy_mid_ratio']=df['act_buy_mid']/df['amount']
    # df['net_power_open']=df['net_buy_open_mean']/df['net_buy_open_std']

    # 开盘后知情主卖占比(占同时段成交额)=-1*(盘后知情主卖额/开盘后成交金额) #已写过 在58里
    # 收盘前知情主买占比(占全天成交额)=-1*(收盘前知情主买额/总成交金额)
    # df1=feather.read_dataframe(r'D:\tyx\检查更新\2021.6-2025.8\buy_close_ratio.feather')
    # df=df.merge(df1,on=['TICKER','DATE'],how='left')
    # df['actbuy_lose']=df['buy_close_ratio']*df['amount_close']
    # # 因子计算3:
    # df['know_buy_close_ratio']=-1*(df['actbuy_lose']/df['amount'])

    # df=df[['DATE','TICKER','big_buy_ratio','buy_concentration','act_buy_mid_ratio','net_power_open','know_buy_close_ratio']]
    df=df[['DATE','TICKER','big_buy_ratio','buy_concentration']]
    # df=df[['DATE','TICKER','know_buy_close_ratio']]
    # print(df)
    return [df]

def update(today='20250905'):
    update_muli('big_buy_ratio.feather',today,run)

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
                print(tmp)
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))
                # feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))

if __name__=='__main__':
    update()
    # run('20250102','20250110')