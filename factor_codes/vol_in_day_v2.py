import feather
import os
import pandas as pd
import warnings
from joblib import Parallel,delayed
from tqdm import tqdm
from tool_tyx.path_data import DataPath
import numpy as np
from tool_tyx.tyx_funcs import update_muli, process_na_stock


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
    # print('df_min:',df_min)
    #
    # ## 平均单笔成交金额类因子--------------------------------------------------------------------------------------
    tmp1=df_min.groupby('TICKER').agg({'amount':'sum','transaction_number':'sum'}).reset_index().rename(columns={'amount':'amount_total','transaction_number':'transaction_number_total'})
    tmp2=df_min[df_min['min_ret']>0].groupby('TICKER').agg({'amount':'sum','transaction_number':'sum'}).reset_index().rename(columns={'amount':'inflow_amount_total','transaction_number':'inflow_num_total'})
    tmp3=df_min[df_min['min_ret']<0].groupby('TICKER').agg({'amount':'sum','transaction_number':'sum'}).reset_index().rename(columns={'amount':'outflow_amount_total','transaction_number':'outflow_num_total'})
    tmp=tmp1.merge(tmp2,on='TICKER',how='inner').merge(tmp3,on='TICKER',how='inner')
    # print('tmp:',tmp)
    tmp['AmtPerTrd'] = tmp['amount_total'] / tmp['transaction_number_total']  # 平均单笔成交金额
    tmp['AmtPerTrd_inFlow']=tmp['inflow_amount_total']/tmp['inflow_num_total'] #平均单笔流入金额
    tmp['AmtPerTrd_outFlow']=tmp['outflow_amount_total']/tmp['outflow_num_total'] #平均单笔流出金额
    tmp['ApT_inFlow_ratio']=tmp['AmtPerTrd_inFlow']/tmp['AmtPerTrd'] #平均单笔流入金额占比
    tmp['ApT_outFlow_ratio']=tmp['AmtPerTrd_outFlow']/tmp['AmtPerTrd'] #平均单笔流出金额占比
    tmp['ApT_netInFlow_ratio']=tmp['ApT_inFlow_ratio']/tmp['ApT_outFlow_ratio'] #平均单笔流入流出金额之比
    df=tmp.copy()[['TICKER','AmtPerTrd','ApT_outFlow_ratio']] #,'ApT_inFlow_ratio','ApT_netInFlow_ratio'
    # print('df:',df)
    ## 大单资金流向类因子-----------------------------------------------------------------------------------
    tmp=df_min[['TICKER', 'min','amount','min_ret','transaction_number']]
    tmp['AmtPerTrd']=tmp['amount']/tmp['transaction_number']
    tmp.sort_values(['TICKER','AmtPerTrd'],ascending=[True,False],inplace=True)
    df2=pd.DataFrame()
    for num in [24,48,70]: #24,48,70
        tt=tmp.groupby('TICKER').head(num) #0.1
        tt.replace([np.inf, -np.inf], np.nan, inplace=True)
        # tmp=tmp.groupby('TICKER').apply(lambda x: x.head(int(len(x)*0.1))) # 太慢
        helper=tt.groupby('TICKER').apply(lambda x: np.prod(1 + x['min_ret'])).reset_index()
        # tmp['Mom_bigOrder']=np.prod(1+tmp['min_ret']*(-1 if tmp['outflow_amount']>tmp['inflow_amount'] else 1))
        # tt1 = tt[tt['min_ret'] > 0].groupby('TICKER').agg({'amount': 'sum'}).reset_index().rename(columns={'amount': 'inflow_amount_total'})
        # tt2 = tt[tt['min_ret'] < 0].groupby('TICKER').agg({'amount': 'sum'}).reset_index().rename(columns={'amount': 'outflow_amount_total'})
        # tt3 = tt.groupby('TICKER').agg({'amount': 'sum'}).reset_index().rename(columns={'amount': 'amount_total'})
        # tt=tt1.merge(tt2,on='TICKER',how='inner').merge(tt3,on='TICKER',how='inner')
        # tt[f'Amt_netInFlow_bigOrder_{num}']=tt['inflow_amount_total']-tt['outflow_amount_total']
        # tt[f'Amt_netInFlow_bigOrder_ratio_{num}']=tt[f'Amt_netInFlow_bigOrder_{num}']/tt['amount_total']
        # tt=tt.merge(helper,on='TICKER',how='inner').rename(columns={0:f'Mom_bigOrder_{num}'})
        helper.rename(columns={0:f'Mom_bigOrder_{num}'},inplace=True)
        if df2.empty:
            df2=helper.copy()
            continue
        df2=df2.merge(helper,on='TICKER',how='inner')
        # print('df2:',df2)
    ## 合并
    # df=df2.copy()
    df=df.merge(df2,on='TICKER',how='inner')
    df['DATE']=date
    # print('df：',df)
    return df.drop_duplicates()

def run(start='20250101',end='20250114'):
    warnings.filterwarnings('ignore')
    calender = pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'))
    calender['trade_date'] = calender['trade_date'].astype(str)
    calender = calender[(calender.trade_date >= start) & (calender.trade_date <= end)]
    tar_date = sorted(list(calender.trade_date.unique()))

    df=Parallel(n_jobs=15)(delayed(get_data_min)(date) for date in tqdm(tar_date,desc='每天因子计算中'))
    df=pd.concat(df).drop_duplicates()
    df=process_na_stock(df,col='AmtPerTrd')
    res=[]
    for col in ['AmtPerTrd', 'ApT_outFlow_ratio', 'Mom_bigOrder_24','Mom_bigOrder_48', 'Mom_bigOrder_70']:
        tmp=df[['TICKER','DATE',col]].drop_duplicates()
        tmp.sort_values(['TICKER','DATE'],inplace=True)
        tmp[col+'_rolling20']=tmp.groupby('TICKER')[col].rolling(20,5).mean().values
        tmp=tmp[['TICKER','DATE',col+'_rolling20']]
        res.append(tmp)
    return res

def update(today='20250904'):
    update_muli('AmtPerTrd_rolling20.feather',today,run,num=-150)
    # pass

if __name__=='__main__':
    update()
    # print(run())
    # start = '20250114'
    # end = '20250117'
    # calender = pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'))
    # calender['trade_date'] = calender['trade_date'].astype(str)
    # calender = calender[(calender.trade_date >= start) & (calender.trade_date <= end)]
    # tar_date = sorted(list(calender.trade_date.unique()))
    # df=Parallel(n_jobs=15)(delayed(get_data_min)(date) for date in tqdm(tar_date,desc='每天因子计算中'))
    # # print(df)
    # print()