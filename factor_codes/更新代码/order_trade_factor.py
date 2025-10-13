# @Author: Yixin Tian
# @File: order_trade_factor.py
# @Date: 2025/9/11 10:18
# @Software: PyCharm
import os
import warnings
import feather
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import get_tar_date, read_min_data, process_na_stock
import numpy as np

def get_data(date):
    warnings.filterwarnings('ignore')
    df=read_min_data(date)[['TICKER','min','active_buy_amount','active_sell_amount','amount']]
    if df.empty:
        return
    try:
        order_sh=feather.read_dataframe(os.path.join(DataPath.order_min_sh,f'{date}.feather'))
        order_sh['order_sell_amount'] = order_sh['Amount_1_order_sell'] + order_sh['Amount_2_order_sell'] + order_sh['Amount_3_order_sell'] + order_sh['Amount_4_order_sell'] + order_sh['Amount_5_order_sell']
        order_sh['order_buy_amount'] = order_sh['Amount_1_order_buy'] + order_sh['Amount_2_order_buy'] + order_sh['Amount_3_order_buy'] + order_sh['Amount_4_order_buy'] + order_sh['Amount_5_order_buy']
        order_sh = order_sh[['TICKER', 'min', 'order_sell_amount', 'order_buy_amount']]
    except FileNotFoundError:
        order_sh=pd.DataFrame()
    try:
        order_sz=feather.read_dataframe(os.path.join(DataPath.order_min_sz,f'{date}.feather'))
        order_sz['order_sell_amount'] = order_sz['Amount_1_order_sell'] + order_sz['Amount_2_order_sell'] + order_sz['Amount_3_order_sell'] + order_sz['Amount_4_order_sell'] + order_sz['Amount_5_order_sell']
        order_sz['order_buy_amount'] = order_sz['Amount_1_order_buy'] + order_sz['Amount_2_order_buy'] + order_sz['Amount_3_order_buy'] + order_sz['Amount_4_order_buy'] + order_sz['Amount_5_order_buy']
        order_sz = order_sz[['TICKER', 'min', 'order_sell_amount', 'order_buy_amount']]
    except FileNotFoundError:
        order_sz=pd.DataFrame()
    order_df=pd.concat([order_sz,order_sh]).reset_index(drop=True)
    if order_df.empty:
        return
    df=df.merge(order_df,on=['TICKER','min'])
    df=df[df['min']<1457]
    # 因子数据准备
    df['net_order_amount']=df['order_buy_amount']-df['order_sell_amount'] # 净委买额
    df.sort_values(['TICKER','min'],inplace=True)
    df['net_order_diff']=df.groupby('TICKER')['net_order_amount'].diff() #净委买变化额
    df['net_trade_amount']=df['active_buy_amount']-df['active_sell_amount'] #净主买成交额
    df['buy_willingness']=df['net_order_diff']+df['net_trade_amount']

    # 计算全天因子
    # tmp1=pd.DataFrame(df.groupby('TICKER')['buy_willingness'].sum()/df.groupby('TICKER')['amount'].sum()).rename(columns={0:'day_buy_will_ratio'})
    # tmp2=pd.DataFrame(df.groupby('TICKER')['buy_willingness'].mean()/df.groupby('TICKER')['buy_willingness'].std()).rename(columns={'buy_willingness':'day_buy_will_str'})
    # tmp=tmp1.merge(tmp2,left_index=True,right_index=True,how='inner')
    # 计算开盘后
    tt=df[(df['min']>=930)&(df['min']<=959)]
    tmp1=pd.DataFrame(tt.groupby('TICKER')['buy_willingness'].sum()/tt.groupby('TICKER')['amount'].sum()).rename(columns={0:'open_buy_will_ratio'})
    tmp2=pd.DataFrame(tt.groupby('TICKER')['buy_willingness'].mean()/tt.groupby('TICKER')['buy_willingness'].std()).rename(columns={'buy_willingness':'open_buy_will_str'})
    tmp=tmp1.merge(tmp2,left_index=True,right_index=True,how='inner').reset_index()
    # 计算盘中
    # tt=df[(df['min']>=1000)&(df['min']<=1426)]
    # tmp1 = pd.DataFrame(tt.groupby('TICKER')['buy_willingness'].sum() / tt.groupby('TICKER')['amount'].sum()).rename(columns={0: 'inday_buy_will_ratio'})
    # tmp2 = pd.DataFrame(tt.groupby('TICKER')['buy_willingness'].mean() / tt.groupby('TICKER')['buy_willingness'].std()).rename(columns={'buy_willingness': 'inday_buy_will_str'})
    # tmp = tmp.merge(tmp1, left_index=True, right_index=True, how='inner').merge(tmp2, left_index=True, right_index=True,how='inner')
    # # 计算收盘前
    # tt = df[(df['min'] >= 1427) & (df['min'] <= 1456)]
    # tmp1 = pd.DataFrame(tt.groupby('TICKER')['buy_willingness'].sum() / tt.groupby('TICKER')['amount'].sum()).rename(columns={0: 'close_buy_will_ratio'})
    # tmp2 = pd.DataFrame(tt.groupby('TICKER')['buy_willingness'].mean() / tt.groupby('TICKER')['buy_willingness'].std()).rename(columns={'buy_willingness': 'close_buy_will_str'})
    # tmp = tmp.merge(tmp1, left_index=True, right_index=True, how='inner').merge(tmp2, left_index=True, right_index=True,how='inner').reset_index()
    # print(tmp)
    # print(tmp.columns)
    tmp['DATE']=date
    # return tmp[['TICKER','DATE','day_buy_will_ratio', 'day_buy_will_str','open_buy_will_ratio', 'open_buy_will_str',
    #             'inday_buy_will_ratio','inday_buy_will_str', 'close_buy_will_ratio', 'close_buy_will_str']]
    return tmp[['TICKER','DATE','open_buy_will_ratio', 'open_buy_will_str']]

def run(start,end):
    warnings.filterwarnings('ignore')
    # get_data('20250828')
    tar_date=get_tar_date(start,end)
    df=Parallel(n_jobs=3)(delayed(get_data)(date) for date in tqdm(tar_date))
    df=pd.concat(df).reset_index(drop=True)
    df=process_na_stock(df,col='open_buy_will_ratio')
    res = []
    for col in df.columns.tolist()[2:]:
        tmp = df[['DATE', 'TICKER', col]]
        tmp.sort_values(['TICKER','DATE'],inplace=True)
        tmp[col + '_roll20'] = tmp.groupby('TICKER')[col].rolling(20, 5).mean().values
        tmp = tmp[['DATE', 'TICKER', col + '_roll20']]
        res.append(tmp)
    return res

def update(today='20250919'):
    update_muli('open_buy_will_ratio_roll20.feather', today, run,num=-70)

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