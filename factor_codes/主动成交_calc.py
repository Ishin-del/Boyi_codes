# @Author: Yixin Tian
# @File: 主动成交.py
# @Date: 2025/9/3 14:18
# @Software: PyCharm
import warnings
import time
import feather
import os

import pandas as pd
import polars as pl
from tqdm import tqdm
from joblib import Parallel,delayed
import numpy as np


class DataPath():
    # 数据存储路径-----------------------------------------------------------------
    factor_out_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子'
    save_path_old=r'D:\tyx\检查更新\2020-2022.12'
    save_path_update=r'D:\tyx\检查更新\2021.6-2025.8'
    save_help_path=r'D:\tyx\raw_data'
    tmp_path=r'D:\tyx\中间数据'

    # 数据路径-------------------------------------------------------
    daily_path=r'\\192.168.1.101\local_data\Data_Storage\daily.feather'
    sh_min=r'\\DESKTOP-79NUE61\SH_min_data' # 上海数据 逐笔合分钟数据
    sz_min=r'\\DESKTOP-79NUE61\SZ_min_data' # 深圳数据 逐笔合分钟数据
    feather_2022 = r'\\192.168.1.28\h\data_feather(2022)\data_feather'
    feather_2023 = r'\\192.168.1.28\h\data_feather(2023)\data_feather'
    feather_2024 = r'\\192.168.1.28\i\data_feather(2024)\data_feather'
    feather_2025 = r'\\192.168.1.28\i\data_feather(2025)\data_feather'
    moneyflow_sh=r'\\DESKTOP-79NUE61\money_flow_sh'
    moneyflow_sz=r'\\DESKTOP-79NUE61\money_flow_sz'
    # moneyflow数据按照4万，20万，100万为分界线，分small,medium,large,xlarge,此数据将集合竞价数据考虑进来了
    mkt_index=r'\\192.168.1.101\local_data\ex_index_market_data\day'
    to_df_path=r'\\192.168.1.101\local_data\Data_Storage' #float_mv,adj_factors的数据路径,citic_code.feather数据
    to_data_path=r'\\192.168.1.101\local_data\base_data' #totalShares数据路径
    to_path=r'\\192.168.1.101\local_data' #calendar.csv数据路径
    wind_A_path=r'\\192.168.1.101\ssd\local_data\Data_Storage\881001.WI.csv' #万得全A指数路径
    feather_sh=r'\\Desktop-79nue61\sh'
    feather_sz=r'\\Desktop-79nue61\sz'

def update_muli(filename,today,run,start,end,num=-50,):
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
                    print('检查更新出错!')
                    exit()
    else:
        print('因子生成中')
        # res=run(start='20200101',end='20221231')
        res=run(start=start,end=end)
        # res=run(start='20250828',end='20250829')
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                print(tmp)
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))

"""
买卖成交额，根据单号合并计算（总成交额，主动成交额）
=> rolling(20) ，20天某只股票的所有数据拉下来
=> 明显的偏度,对数调整后，计算均值和标准差
=> 设置阈值：
    大买（卖）单：大于均值+1倍标准差
    中买（卖）单：大于均值且小于等于均值+1倍标准差
    小买（卖）单：小于均值
-------------------------------------------------------------
大买（卖）单主动成交度=大买（卖）单主动成交金额/大买（卖）单成交金额
中买（卖）单主动成交度=中买（卖）单主动成交金额/中买（卖）单成交金额
小买（卖）单主动成交度=小买（卖）单主动成交金额/小买（卖）单成交金额
"""

def get_tick_data(date):
    # print('!')
    # t1=time.time()
    try:
        tick_sh=feather.read_dataframe(os.path.join(DataPath.feather_sh,date+'_trade_sh.feather'),columns=['TICKER','BuyOrderNo', 'SellOrderNo','TradeMoney'])
    except FileNotFoundError:
        tick_sh=pd.DataFrame()
    try:
        tick_sz = feather.read_dataframe(os.path.join(DataPath.feather_sz, date + '_trade_sz.feather'),columns=['TICKER','BuyOrderNo', 'SellOrderNo','TradeMoney'])
    except FileNotFoundError:
        tick_sz = pd.DataFrame()
    # tick_sz = pd.DataFrame()
    res=pd.DataFrame()
    # print('!')
    for tick_df in [tick_sh,tick_sz]:
        if tick_df.empty:
            continue
        tick_df=pl.from_pandas(tick_df)
        # 根据同单号 计算成交金额
        tmp_buy=tick_df.group_by(['TICKER', 'BuyOrderNo']).agg(pl.col('TradeMoney').sum().alias('buy_money'))
        tmp_sell=tick_df.group_by(['TICKER', 'SellOrderNo']).agg(pl.col('TradeMoney').sum().alias('sell_money'))
        # 根据同单号 计算主动成交金额
        tmp_act_buy=tick_df.filter(pl.col('BuyOrderNo')>pl.col('SellOrderNo')).group_by(['TICKER', 'BuyOrderNo']).agg(pl.col('TradeMoney').sum().alias('buy_act_money'))
        tmp_act_sell=tick_df.filter(pl.col('BuyOrderNo')<pl.col('SellOrderNo')).group_by(['TICKER', 'SellOrderNo']).agg(pl.col('TradeMoney').sum().alias('sell_act_money'))
        tmp_buy=tmp_buy.join(tmp_act_buy, on=['TICKER', 'BuyOrderNo'], how='inner')
        tmp_sell=tmp_sell.join(tmp_act_sell, on=['TICKER', 'SellOrderNo'], how='inner')
        tmp1=cal(tmp_buy,'buy',date)
        tmp2=cal(tmp_sell,'sell',date)
        tmp=tmp1.join(tmp2, on=['DATE','TICKER'], how='inner')
        tmp= tmp.with_columns(pl.lit(date).alias('DATE'))
        if len(res) == 0:
        # if res.empty:
            res = tmp
        else:
            res = pl.concat([res, tmp]).to_pandas()
    # print(res)
    # t2=time.time()
    # print(f'运行时间：{str(t2-t1)}')
    return res

def cal(tmp_df,label,date):
    warnings.filterwarnings('ignore')
    # 同只股票取对数
    tmp_df = tmp_df.group_by('TICKER').agg(
        [pl.col(label+'_money').alias(label+'_money'), pl.col(label+'_act_money').alias(label+'_act_money'),
         pl.col(label+'_money').log().alias('log_'+label), pl.col(label+'_act_money').log().alias('log_act_'+label)]).explode(
        [label+'_money', label+'_act_money', 'log_'+label, 'log_act_'+label])
    # 算均值标准差
    help = tmp_df.group_by('TICKER').agg(pl.col('log_'+label).mean().alias('money_mean'),
                                          pl.col('log_'+label).std().alias('money_std'))
    tmp_df = tmp_df.join(help, on=['TICKER'], how='left')
    tmp_df = tmp_df.with_columns((pl.col('money_mean') + pl.col('money_std')).alias('mean_std'))
    # 判断大中小单
    tmp_df = tmp_df.with_columns(pl.when(pl.col('log_'+label) >= (pl.col('mean_std'))).then(pl.lit('big'))
        .when((pl.col('log_'+label) < pl.col('mean_std')) & (pl.col('log_'+label) >= pl.col('money_mean'))).then(
        pl.lit('med')).otherwise(pl.lit('small')).alias('order_size'))
    # 计算因子
    tmp_df = tmp_df.group_by(['TICKER', 'order_size']).agg(pl.col(label+'_money').sum().alias(label+'_total'),
                                                             pl.col(label+'_act_money').sum().alias('act_total'))
    tmp_df = tmp_df.with_columns((pl.col('act_total') / pl.col(label+'_total')).alias('act_deal_degree'))
    tmp_df = tmp_df.pivot(columns='order_size', index='TICKER', values='act_deal_degree')
    tmp_df = tmp_df.with_columns(pl.lit(date).alias('DATE'))
    tmp_df = tmp_df.rename({'small': 'small_degree_' + label,'med': 'med_degree_' + label,'big': 'big_degree_' + label})
    tmp_df = tmp_df.select(['DATE','TICKER','small_degree_' + label,'med_degree_' + label,'big_degree_' + label])
    # print(tmp_df)
    return tmp_df

def run(start='20250828',end='20250829'):
    calender=pd.read_csv(os.path.join(DataPath.to_path,'calendar.csv'))
    calender['trade_date']=calender['trade_date'].astype(str)
    calender=calender[(calender.trade_date>=start) &(calender.trade_date<=end)]
    tar_date=sorted(list(calender.trade_date.unique()))
    # tmp = Parallel(n_jobs=2)(delayed(get_tick_data)(date) for date in tqdm(tar_date))
    tmp=[]
    for date in tqdm(tar_date):
        tt=get_tick_data(date)
        tmp.append(tt)
    tmp=pd.concat(tmp)
    # print(tmp)
    return [tmp]

def update(today='20250822'):
    update_muli('small_degree_buy.feather',today,run,start='20200102',end='20250822',num=-10)

if __name__=='__main__':
    update()
    # run()