import feather
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from tool_tyx.path_data import DataPath
import os
import warnings
from tqdm import tqdm

from tool_tyx.tyx_funcs import process_na_stock


def get_data_min(date):
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
        df_min=df_min[['TICKER', 'min', 'open', 'high', 'low', 'close', 'volume','DATE']]
    # todo:
    # df_min=df_min[(df_min.TICKER=='603377.SH')|(df_min.TICKER=='000001.SZ')]
    df_min=df_min[(df_min['min']!=930)&(df_min['min']<1457)] #正常应该是237分钟数据
    df_sig=df_min.groupby('TICKER').agg({'min':len})
    # if date=='20221206':
    #     tar_codes = df_sig[df_sig['min'] >= 235].index
    # else:
    #     tar_codes=df_sig[df_sig['min'] == 237].index
    tar_codes = df_sig[df_sig['min'] >= 235].index
    df_min = df_min[df_min['TICKER'].isin(tar_codes)]
    if df_min.empty:
        return
    df_min.sort_values(['TICKER','min'],inplace=True)
    factor_df1=cal1(df_min)
    factor_df2=cal2(df_min)
    return factor_df1,factor_df2

def cal2(df_min):
    """弹性系数较大的股票，可以理解为流动性相对较好，对短期交易放量的反应相对较小，弹性系数较小的票，流动性相对较弱，对短期交易
        量的冲击反应较大。"""
    warnings.filterwarnings('ignore')
    date=df_min.DATE.iloc[0]

    df_min['vol_mean_5min']=df_min.groupby('TICKER').rolling(5)['volume'].mean().values
    df_min['price_volatility']=(df_min['high']-df_min['low'])/df_min['open']
    # df_min['flag']=np.where((df_min['volume']-df_min['vol_mean_5min'])>df_min['vol_mean_5min'],1,0) #1:激增时刻,0:普通时刻
    df_min['flag']=np.where(df_min['volume']>2*df_min['vol_mean_5min'],'rush_mean','normal_mean') #1:激增时刻,0:普通时刻
    tmp=df_min.groupby(['TICKER','flag'])['price_volatility'].mean().reset_index()
    tmp=tmp.pivot(index='TICKER', columns='flag', values='price_volatility').reset_index() #.rename(columns={0:'normal_mean',1:'rush_mean'})
    tmp['elasticity']=1-tmp['rush_mean']/tmp['normal_mean']
    tmp.replace([np.inf,-np.inf],np.nan,inplace=True)
    tmp['elasticity_adjust']=np.abs(tmp['elasticity']-tmp['elasticity'].mean())
    tmp['DATE']=date
    return tmp[['DATE','TICKER','elasticity_adjust']]

# 于股票日内交易量的分布特征,香农熵（Shannon Entropy）刻画，熵越大，个股日内相对交易量的分布越趋于均匀分布，否则，股票的交易越有可能由信息驱动
# todo: 237分钟/5分钟
def cal1(df_min):
    warnings.filterwarnings('ignore')
    date=df_min['DATE'].iloc[0]
    vol_total_min = df_min.groupby('min')['volume'].sum().reset_index().rename(columns={'volume': 'volume_total_min'})
    # 全市场所有股票成交量
    df_min = df_min.merge(vol_total_min, on='min', how='left')  # 每分钟个股的相对成交量
    df_min['rela_min_vol'] = df_min['volume'] / df_min['volume_total_min'] #每分钟个股的相对成交量（个股成交量/全市场所有股票成交量
    # todo:
    vol_total_min_rela = df_min.groupby('TICKER')['rela_min_vol'].sum().reset_index().rename(columns={'rela_min_vol': 'vol_min_rela_sum'})
    # 同只股票 全部时间区间 相对成交量之和
    df_min=df_min.merge(vol_total_min_rela,on='TICKER',how='left')
    df_min['time_group'] =df_min.groupby('TICKER').cumcount() // 5
    tmp=df_min.groupby(['TICKER','time_group'])['rela_min_vol'].sum().reset_index().rename(columns={'rela_min_vol':'total_rela_vol_5min'})
    # 3分钟区间内 相对成交量 之和
    df_min=df_min.merge(tmp,on=['TICKER','time_group'],how='left')
    df_min['ratio_5min']=df_min['total_rela_vol_5min']/df_min['vol_min_rela_sum'] #3分钟占比p(xi)
    tmp=df_min[['TICKER','time_group','ratio_5min']].drop_duplicates()
    # print(tmp)
    df=tmp.groupby('TICKER')['ratio_5min'].agg(lambda x: -np.sum(x * np.log2(x))).reset_index().rename(columns={'ratio_5min':'entropy'}) #日度成交量分布熵值因子
    df['DATE']=date
    df['entropy_adjust']=np.abs(df['entropy']-df['entropy'].mean())
    # print(df[['DATE','TICKER','entropy_adjust']])
    return df[['DATE','TICKER','entropy_adjust']]

def run(start='20200101',end='20250822'):
    # daily_df=feather.read_dataframe(DataPath.daily_path)
    calender=pd.read_csv(os.path.join(DataPath.to_path,'calendar.csv'))
    calender['trade_date']=calender['trade_date'].astype(str)
    calender=calender[(calender.trade_date>=start) &(calender.trade_date<=end)]
    tar_date=sorted(list(calender.trade_date.unique()))
    # todo:对get_data_min做多线程
    # print(tar_date)
    tmp=Parallel(n_jobs=15)(delayed(get_data_min)(date) for date in tqdm(tar_date))
    # tmp=get_data_min('20221206')
    tmp=[x for x in tmp if x is not None]
    factor1=[x[0] for x in tmp]
    factor2=[x[1] for x in tmp]
    factor1=pd.concat(factor1)
    factor2=pd.concat(factor2)

    factor1 = process_na_stock(factor1,col='entropy_adjust')
    factor2 = process_na_stock(factor2,col='elasticity_adjust')

    factor1.sort_values(['TICKER','DATE'],inplace=True)
    factor1['entropy_adjust_mean20']=factor1.groupby('TICKER')['entropy_adjust'].rolling(20,5).mean().values
    factor1['entropy_adjust_std20']=factor1.groupby('TICKER')['entropy_adjust'].rolling(20,5).std().values
    factor1['vol_entropy']=(factor1['entropy_adjust_std20']+factor1['entropy_adjust_std20'])/2
    factor1.drop(columns='entropy_adjust',inplace=True)

    factor2.sort_values(['TICKER', 'DATE'], inplace=True)
    factor2['elasticity_adjust_mean20']=factor2.groupby('TICKER')['elasticity_adjust'].rolling(20,5).mean().values
    factor2['elasticity_adjust_std20']=factor2.groupby('TICKER')['elasticity_adjust'].rolling(20,5).std().values
    factor2['flow_elasticity']=(factor2['elasticity_adjust_mean20']+factor2['elasticity_adjust_std20'])/2
    factor2.drop(columns='elasticity_adjust',inplace=True)

    factor2=factor1[['DATE','TICKER','vol_entropy']].merge(factor2,on=['DATE','TICKER'],how='right')
    factor2['暗流涌动']=(factor2['vol_entropy']+factor2['flow_elasticity'])/2
    factor2.drop(columns='vol_entropy',inplace=True)
    return factor1,factor2

def update(today='20250822'):
    if os.path.exists(os.path.join(DataPath.save_path_update,'entropy_adjust_mean20.feather')):
        old=feather.read_dataframe(os.path.join(DataPath.save_path_update,'entropy_adjust_mean20.feather'))
        new_start=sorted(old.DATE.drop_duplicates().to_list())[-50]
        df1,df2=run(start=new_start,end=today)
        # print('!')
        # df1,df2=run(start='20221206', end='20221206')
        for df in [df1,df2]:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                # print(tmp)
                old=feather.read_dataframe(os.path.join(DataPath.save_path_update,col+'.feather'))
                # old = pd.concat([old, tmp]).reset_index(drop=True).drop_duplicates()
                # feather.write_dataframe(old, os.path.join(DataPath.save_path_update, col + '.feather'))
                # feather.write_dataframe(old, os.path.join(DataPath.factor_out_path, col + '.feather'))
                test=old.merge(tmp,on=['DATE','TICKER'],how='inner').dropna() #.groupby('TICKER').tail(5)
                tar_list = sorted(list(test.DATE.unique()))[-5:]
                test = test[test.DATE.isin(tar_list)]
                if np.isclose(test.iloc[:,2],test.iloc[:,3]).all():
                    tmp=tmp[tmp.DATE>old.DATE.max()]
                    old=pd.concat([old,tmp]).reset_index(drop=True)
                    print(old)
                    feather.write_dataframe(old,os.path.join(DataPath.save_path_update,col+'.feather'))
                    feather.write_dataframe(old,os.path.join(DataPath.factor_out_path,col+'.feather'))
                else:
                    print(test[~np.isclose(test.iloc[:,2],test.iloc[:,3])])
                    print('检查更新出错!')
                    exit()
    else:
        print('因子生成中')
        df1,df2=run(start='20200101',end='20250822')
        # df1,df2=run(start='20220101',end='20250822')
        for df in [df1,df2]:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))
                # feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))


if __name__=='__main__':
    # df1,df2=run(start='20200101',end='20250822')
    # for df in [df1, df2]:
    #     for col in df.columns[2:]:
    #         tmp = df[['DATE', 'TICKER', col]]
    #         feather.write_dataframe(tmp, os.path.join(r'C:\Users\admin\Desktop', col + '.feather'))
    # print('!')
    update()