import feather
import os
import pandas as pd
import warnings
from joblib import Parallel,delayed
from tqdm import tqdm
from tool_tyx.path_data import DataPath
import numpy as np
from tool_tyx.tyx_funcs import update_muli

def get_data_tick(date):
    warnings.filterwarnings('ignore')
    if os.path.exists(os.path.join(DataPath.tmp_path,'流入出分钟金额数据',date+'.feather')):
        return
    df=pd.DataFrame()
    try:
        tick_sh=feather.read_dataframe(os.path.join(DataPath.feather_sh,date+'_trade_sh.feather'),
                                       columns=['TICKER','TickTime','BuyOrderNo', 'SellOrderNo','TradeMoney'])
    except FileNotFoundError:
        tick_sh=pd.DataFrame()
    try:
        tick_sz = feather.read_dataframe(os.path.join(DataPath.feather_sz, date + '_trade_sz.feather'),
                                         columns=['TICKER','TickTime','BuyOrderNo', 'SellOrderNo','TradeMoney'])
    except FileNotFoundError:
        tick_sz = pd.DataFrame()
    # tick_df=pd.concat([tick_sh,tick_sz]).reset_index(drop=True)
    for tick_df in [tick_sh,tick_sz]:
        tick_df['min']=tick_sh['TickTime']//100000
        tick_df['flow_dir']=np.where(tick_df['BuyOrderNo']>tick_df['SellOrderNo'],1,0)
        # 1 : 买，代表流入
        # tick_df
        tmp=tick_df.groupby(['TICKER', 'min']).agg({'flow_dir': ['sum', 'count']}).reset_index()
        tmp.columns = ['TICKER', 'min', 'inflow_num', 'total_num']
        tmp['outflow_num']=tmp['total_num']-tmp['inflow_num']
        tmp.drop(columns='total_num',inplace=True)
        tick_df=tick_df.groupby(['TICKER','min','flow_dir'])['TradeMoney'].sum().reset_index()
        tick_df=(tick_df.pivot(values='TradeMoney',columns='flow_dir',index=['TICKER','min']).reset_index().
                 rename(columns={0:'outflow_amount',1:'inflow_amount'}))
        tick_df=tick_df.merge(tmp,on=['TICKER','min'],how='inner')
        tick_df['DATE']=date
        # print(tick_df)
        df=pd.concat([df,tick_df])
    # print(df)
    os.makedirs(os.path.join(DataPath.tmp_path,'流入出分钟金额数据'),exist_ok=True)
    feather.write_dataframe(df,os.path.join(DataPath.tmp_path,'流入出分钟金额数据',date+'.feather'))

def get_data_min(date):
    warnings.filterwarnings('ignore')
    file=date+'.feather'
    try:
        sh_min=feather.read_dataframe(os.path.join(DataPath.sh_min,file))
    except FileNotFoundError:
        sh_min=pd.DataFrame()
    try:
        sz_min=feather.read_dataframe(os.path.join(DataPath.sz_min,file))
    except FileNotFoundError:
        sz_min=pd.DataFrame()
    df_min=pd.concat([sh_min,sz_min]).reset_index(drop=True)
    if df_min.empty:
        return
    else:
        df_min = df_min[['TICKER', 'min','amount', 'close', 'volume', 'DATE','transaction_number']]
    df_min.sort_values(['TICKER','min'],inplace=True)
    df_min=df_min[(df_min['min']<1457) &(df_min['min']!=930)]
    df_min['min_ret']=df_min.groupby('TICKER')['close'].pct_change()

    ## 平均单笔成交金额类因子--------------------------------------------------------------------------------------
    flow_df=feather.read_dataframe(os.path.join(DataPath.tmp_path,'流入出分钟金额数据',date+'.feather'))
    df_min=df_min.merge(flow_df,on=['min','TICKER','DATE'],how='left')
    tmp=df_min.groupby('TICKER').agg({'inflow_amount':'sum','inflow_num':'sum','outflow_amount':'sum','outflow_num':'sum','amount':'sum','transaction_number':'sum'}).reset_index()
    tmp.columns=['TICKER','inflow_amount_total','inflow_num_total','outflow_amount_total','outflow_num_total','amount_total','transaction_number_total']
    tmp['AmtPerTrd'] = tmp['amount_total'] / tmp['transaction_number_total']  # 平均单笔成交金额
    tmp['AmtPerTrd_inFlow']=tmp['inflow_amount_total']/tmp['inflow_num_total'] #平均单笔流入金额
    tmp['AmtPerTrd_outFlow']=tmp['outflow_amount_total']/tmp['outflow_num_total'] #平均单笔流出金额
    tmp['ApT_inFlow_ratio']=tmp['AmtPerTrd_inFlow']/tmp['AmtPerTrd'] #平均单笔流入金额占比
    tmp['ApT_outFlow_ratio']=tmp['AmtPerTrd_outFlow']/tmp['AmtPerTrd'] #平均单笔流出金额占比
    tmp['ApT_netInFlow_ratio']=tmp['ApT_inFlow_ratio']/tmp['ApT_outFlow_ratio'] #平均单笔流入流出金额之比
    df=tmp.copy()[['TICKER','AmtPerTrd','ApT_inFlow_ratio','ApT_outFlow_ratio','ApT_netInFlow_ratio']]
    ## 大单资金流向类因子----------------------------------------------------------------------------------------
    tmp=df_min[['TICKER', 'min','inflow_amount','outflow_amount','amount','min_ret','transaction_number']]
    tmp['AmtPerTrd']=tmp['amount']/tmp['transaction_number']
    tmp.sort_values(['TICKER','AmtPerTrd'],ascending=[True,False],inplace=True)
    df2=pd.DataFrame()
    for num in [24,48,70]:
        tt=tmp.groupby('TICKER').head(num) #0.1
        # tmp=tmp.groupby('TICKER').apply(lambda x: x.head(int(len(x)*0.1))) # 太慢
        helper=tt.groupby('TICKER').apply(lambda x: np.prod(1 + x['min_ret'])).reset_index()
        # tmp['Mom_bigOrder']=np.prod(1+tmp['min_ret']*(-1 if tmp['outflow_amount']>tmp['inflow_amount'] else 1))
        tt=tt.groupby('TICKER').agg({'inflow_amount':'sum','outflow_amount':'sum','amount':'sum'}).reset_index()
        tt.columns=['TICKER','inflow_amount_total','outflow_amount_total','amount_total']
        tt[f'Amt_netInFlow_bigOrder_{num}']=tt['inflow_amount_total']-tt['outflow_amount_total']
        tt[f'Amt_netInFlow_bigOrder_ratio_{num}']=tt[f'Amt_netInFlow_bigOrder_{num}']/tt['amount_total']
        tt=tt.merge(helper,on='TICKER',how='inner').rename(columns={0:f'Mom_bigOrder_{num}'})
        tt=tt[['TICKER',f'Amt_netInFlow_bigOrder_ratio_{num}',f'Mom_bigOrder_{num}']]
        if df2.empty:
            df2=tt.copy()
            continue
        df2=df2.merge(tt,on='TICKER',how='inner')
    ## 合并
    df=df.merge(df2,on='TICKER',how='inner')
    df['DATE']=date
    return df

def run(start='20250828',end='20250829'):
    # print('.........!')
    warnings.filterwarnings('ignore')
    # daily_df = feather.read_dataframe(DataPath.daily_path)[['DATE','TICKER']]
    # daily_df=daily_df[(daily_df.DATE>=start)&(daily_df.DATE<=end)]
    # print(daily_df)
    calender = pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'))
    calender['trade_date'] = calender['trade_date'].astype(str)
    calender = calender[(calender.trade_date >= start) & (calender.trade_date <= end)]
    tar_date = sorted(list(calender.trade_date.unique()))
    Parallel(n_jobs=15)(delayed(get_data_tick)(date) for date in tqdm(tar_date,desc='计算每分钟流入流出金额数据'))
    df=Parallel(n_jobs=15)(delayed(get_data_min)(date) for date in tqdm(tar_date,desc='每天因子计算中'))
    df=pd.concat(df)
    res=[]
    for col in df.columns.tolist()[1:-1]:
        tmp=df[['TICKER','DATE',col]]
        tmp.sort_values(['TICKER','DATE'],inplace=True)
        tmp[col+'_rolling20']=tmp.groupby('TICKER')[col].rolling(20,5).mean().values
        tmp=tmp[['TICKER','DATE',col+'_rolling20']]
        res.append(tmp)
    return res

def update(today='20250822'):
    update_muli('AmtPerTrd_rolling20.feather',today,run)
    # pass

if __name__=='__main__':
    update()