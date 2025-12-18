# @Author: Yixin Tian
# @File: 主动成交.py
# @Date: 2025/9/3 14:18
# @Software: PyCharm
import warnings
import time
import feather
import os
from tool_tyx.path_data import DataPath
import pandas as pd
import polars as pl
from tqdm import tqdm
from joblib import Parallel,delayed

from tool_tyx.tyx_funcs import update_muli

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
    update_muli('small_degree_buy.feather',today,run,num=-10)

if __name__=='__main__':
    # update()
    # run()
    ll=os.listdir(r'C:\Users\admin\Desktop\test')
    for f in ll:
        df=feather.read_dataframe(os.path.join(r'C:\Users\admin\Desktop\test',f))
        df.sort_values(['TICKER','DATE'],inplace=True)
        df[f.replace('.feather','')+'_roll20']=df.groupby('TICKER')[f.replace('.feather','')].rolling(20,5).mean().values
        df=df[['DATE','TICKER',f.replace('.feather','')+'_roll20']]
        feather.write_dataframe(df,os.path.join(r'C:\Users\admin\Desktop\test',f.replace('.feather','')+'_roll20.feather'))
        # break
        print(df)