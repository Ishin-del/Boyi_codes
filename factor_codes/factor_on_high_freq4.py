import math
import os
import warnings
import feather
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

class DataPath():
    # 数据存储路径-----------------------------------------------------------------
    factor_out_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子'
    save_path_old=r'D:\tyx\检查更新\2020-2022.12'
    save_path_update=r'D:\tyx\检查更新\2021.6-2025.8'
    save_help_path=r'D:\tyx\raw_data'
    tmp_path=r'D:\tyx\中间数据'
    # tmp_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用'

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
    to_df_path=r'\\192.168.1.101\local_data\Data_Storage' #float_mv.feather,adj_factors(复权因子)数据路径,citic_code.feather数据(行业数据),money_flow
    to_data_path=r'\\192.168.1.101\local_data\base_data' #totalShares数据路径
    to_path=r'\\192.168.1.101\local_data' #calendar.csv数据路径
    wind_A_path=r'\\192.168.1.101\ssd\local_data\Data_Storage\881001.WI.csv' #万得全A指数路径
    feather_sh=r'\\Desktop-79nue61\sh'
    feather_sz=r'\\Desktop-79nue61\sz'

    order_min_sh=r'\\DESKTOP-79NUE61\SH_min_data_big_order'
    order_min_sz=r'\\DESKTOP-79NUE61\SZ_min_data_big_order'

    ind_df_path=r'\\192.168.1.101\local_data\Data_Storage\citic_code.feather'
    # -------------------------------------------
    # 机器学习数据路径
    train_data_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\rolling_generation_202508_sz_auction_end_20240815_908\train'
    train_big_order_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\rolling_generation_202508_sz_auction_end_20240815_909_big_order\train'

    # 最终因子路径
    factor_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子'
    ret_df_path = r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\return_因子组合用.feather'
# 20240114-中信建投-流动性高频因子再构建与投资者注意力因子
def get_tar_date(start,end):
    calender = pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'))
    calender['trade_date'] = calender['trade_date'].astype(str)
    calender = calender[(calender.trade_date >= start) & (calender.trade_date <= end)]
    tar_date = sorted(list(calender.trade_date.unique()))
    return tar_date

def process_na_stock(df,col):
    '''解决某些股票出现时间序列断开的情况，这种情况会在rolling（window!=min_period）的时候,因子值出现不同'''
    '''重组再上市情况'''
    pt = df.pivot_table(columns='TICKER', index='DATE', values=col)
    pt = pt.ffill()
    pp = pt.melt(ignore_index=False).reset_index(drop=False)
    df = pd.merge(pp, df, how='left', on=['TICKER', 'DATE'])
    use_cols = list(np.setdiff1d(df.columns, ['TICKER', 'DATE']))
    for c in use_cols:
        df[c] = df.groupby('TICKER')[c].ffill()
    df.drop(columns='value', inplace=True)
    return df

def cal(code,date='20251029',path='',snap_time=3):
    warnings.filterwarnings('ignore')
    # date='20251029'
    # df=feather.read_dataframe(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\orderbook\20251029\000001.feather')
    df=feather.read_dataframe(os.path.join(path,code))
    df.sort_values('TickTime', inplace=True)
    df['DateTime'] = df['TickTime'] // 1000
    df = df[(df['DateTime'] >= int(date + '093000')) & (df['DateTime'] < int(date + '145800'))].reset_index(
        drop=True)
    df.drop_duplicates(subset='DateTime', keep='last', inplace=True)
    df = df.reset_index(drop=True).iloc[::snap_time]
    df['pre_LastPrice']=df['LastPrice'].shift(1)
    df['dir']=np.where(df['LastPrice']>df['pre_LastPrice'],1,-1) # bid买1 ask卖-1
    df[f'price_mean'] = (df[f'Bid_0'] + df[f'Offer_0']) / 2
    # ES指标---------------------
    df[f'ES'] = 2 * df['dir'] * (df['LastPrice'] - df[f'price_mean']) / df[f'price_mean']
    df.sort_values('DateTime',inplace=True)
    df['price_mean_t_5']=df['price_mean'].shift(-5) #用shift(-5)了,因为研报里是t+5
    # PRS指标---------------------
    df[f'PRS'] = 2 * df['dir'] * (np.log(df['LastPrice']) - np.log(df[f'price_mean_t_5']))
    res1=df['ES'].mean()
    res2=df['PRS'].mean()
    # CPQS指标
    df_tail_1=df.tail(1)
    df['CPQS']=(df_tail_1['Offer_0']-df_tail_1['Bid_0'])/df_tail_1['price_mean']
    res3=df['CPQS'].mean()
    # 高频，ESI指标,PRSI指标,CPQSI指标在日频里计算了
    return [date,code,res1,res2,res3] #ES,PRS,CPQS

def run(start,end,time_len=20):
    warnings.filterwarnings('ignore')
    # tar_date=get_tar_date(start,end)
    # df=[]
    # for date in tar_date:
    #     if date[:4] == '2022':
    #         path = os.path.join(DataPath.feather_2022, date, 'Snapshot')
    #         code_files = os.listdir(path)
    #     elif date[:4] == '2023':
    #         path = os.path.join(DataPath.feather_2023, date, 'Snapshot')
    #         code_files = os.listdir(path)
    #     elif date[:4] == '2024':
    #         path = os.path.join(DataPath.feather_2024, date, 'Snapshot')
    #         code_files = os.listdir(path)
    #     elif date[:4] == '2025':
    #         path = os.path.join(DataPath.feather_2025, date, 'Snapshot')
    #         code_files = os.listdir(path)
    #     # code_files=os.listdir(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\orderbook\20251029')
    #     # for code in code_files:
    #     #     df=read_snap_data(code)
    #     tmp = Parallel(n_jobs=15)(delayed(cal)(code, path, date) for code in tqdm(code_files,desc=f'{date}数据计算中'))
    #     tmp = pd.concat(tmp).reset_index(drop=True)
    #     tmp.columns = ['DATE', 'TICKER', 'ES', 'PRS', 'CPQS']
    #     df.append(tmp)
    # df = pd.concat(df).reset_index(drop=True)
    # print('---------------------')
    daily_df=feather.read_dataframe(DataPath.daily_path)
    daily_df.sort_values(['TICKER','DATE'],inplace=True)
    daily_df['daily_ret']=daily_df.groupby('TICKER')['close'].pct_change()
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    daily_df.dropna(inplace=True)
    daily_df=process_na_stock(daily_df,'close')
    daily_df=daily_df[(daily_df['DATE']>=start)&(daily_df['DATE']<=end)]
    # daily_df=df.merge(daily_df,on=['DATE','TICKER'],how='right')
    # for col in ['ES','PRS','CPQS']:
    #     daily_df[f'{col}I']=daily_df[col]/daily_df['amount']
    adj_factors=feather.read_dataframe(os.path.join(DataPath.to_df_path,'adj_factors.feather'))
    daily_df=daily_df.merge(adj_factors,on=['TICKER','DATE'],how='left')
    daily_df['close']*=daily_df['adj_factors'] # close得乘复权因子
    daily_df['ILLIQ']=np.abs(daily_df['daily_ret'])/daily_df['amount']
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    # daily_df.dropna(inplace=True)
    daily_df=process_na_stock(daily_df,'close')
    daily_df['Amivest']=daily_df['amount']/np.abs(daily_df['daily_ret'])
    daily_df['log_high_low']=np.log(daily_df['high']/daily_df['low'])
    daily_df['pre_log_high_low']=daily_df.groupby('TICKER')['log_high_low'].shift(1)
    daily_df['pre_high']=daily_df.groupby('TICKER')['high'].shift(1)
    daily_df['pre_low']=daily_df.groupby('TICKER')['low'].shift(1)
    daily_df['max_high']=daily_df[['high','low','pre_high','pre_low']].max(axis=1)
    daily_df['min_low']=daily_df[['high','low','pre_high','pre_low']].min(axis=1)
    daily_df['gamma']=(np.log(daily_df['max_high']/daily_df['min_low']))**2
    daily_df['beta']=((daily_df['log_high_low'])**2+(daily_df['pre_log_high_low'])**2)/2
    daily_df['alpha']=(np.sqrt(2*daily_df['beta'])-np.sqrt(daily_df['beta']))/(3-2*np.sqrt(2))-np.sqrt(daily_df['gamma']/(3-2*np.sqrt(2)))
    daily_df['HL']=2*(np.exp(daily_df['alpha'])-1)/(1+np.exp(daily_df['alpha']))
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    # daily_df.dropna(inplace=True)
    daily_df=process_na_stock(daily_df,'close')
    daily_df['HLI']=daily_df['HL']/daily_df['amount']
    mkt_df=pd.read_csv(DataPath.wind_A_path) # 市场收益率用的wind全A
    mkt_df['mkt_ret']=mkt_df['close'].pct_change()
    mkt_df['DATE']=mkt_df['DATE'].astype(str)
    daily_df=daily_df.merge(mkt_df[['DATE','mkt_ret']],on='DATE',how='left')
    daily_df.sort_values(['TICKER','DATE'],inplace=True)
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    # daily_df.dropna(inplace=True)
    daily_df=process_na_stock(daily_df,'close')
    # 注意力指标
    daily_df['ABNRETED_daily']=daily_df['daily_ret']-daily_df['mkt_ret'] #np.abs(
    daily_df['ABNRETED']=np.abs(daily_df.groupby('TICKER')['ABNRETED_daily'].rolling(time_len,5).max().values)
    daily_df['daily_ret_20']=daily_df.groupby('TICKER')['close'].pct_change(20)
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    # daily_df.dropna(inplace=True)
    daily_df=process_na_stock(daily_df,'close')
    daily_df['ABNRETM']=np.abs(daily_df['daily_ret_20']-daily_df['mkt_ret'])
    daily_df['volume_roll225']=daily_df.groupby('TICKER')['volume'].rolling(225,80).mean().values
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    # daily_df.dropna(inplace=True)
    daily_df=process_na_stock(daily_df,'close')
    daily_df['ABNVOLD_daily']=daily_df['volume']/daily_df['volume_roll225']
    daily_df['ABNVOLD']=np.abs(daily_df.groupby('TICKER')['ABNVOLD_daily'].rolling(20,5).max().values)
    daily_df['volume_month']=daily_df.groupby('TICKER')['volume'].rolling(20,5).sum().values
    # daily_df['volume_roll225']=daily_df.groupby('TICKER')['volume'].rolling(225,80).mean().values
    daily_df['ABNVOLM']=np.abs(daily_df['volume_month']/daily_df['volume_roll225'])
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    daily_df.dropna(inplace=True)
    daily_df=process_na_stock(daily_df,'close')
    daily_df.sort_values(['TICKER','DATE'],inplace=True)
    daily_df['ATTN']=daily_df.groupby('TICKER')['volume'].ewm(span=20).mean().values
    daily_df['daily_ret_225']=daily_df.groupby('TICKER')['close'].pct_change(225)
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    # daily_df.dropna(inplace=True)
    daily_df=process_na_stock(daily_df,'close')
    daily_df['ER']=daily_df['daily_ret_20']/daily_df['daily_ret_225']
    # 凸显理论因子
    daily_df=help(daily_df,'daily_ret','mkt_ret')
    #ret换流通换手率
    float_mv=feather.read_dataframe(os.path.join(DataPath.to_df_path,'float_mv.feather'))
    daily_df=daily_df.merge(float_mv,on=['DATE','TICKER'],how='left')
    daily_df['turnover']=daily_df['amount']/daily_df['float_mv']
    daily_df['turnover_mkt'] = daily_df.groupby('DATE')['turnover'].transform('mean')
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    # daily_df.dropna(inplace=True)
    daily_df=process_na_stock(daily_df,'close')
    daily_df = help(daily_df, 'turnover', 'turnover_mkt')
    # daily_df.to_csv(r'C:\Users\admin\Desktop\check.csv', index_col=0)
    print(daily_df)
    res=daily_df[['DATE','TICKER','ES','PRS','CPQS','ESI','PRSI','CPQSI','ILLIQ','Amivest','HL','HLI','ABNRETED','ABNRETM',
              'ABNVOLD','ABNVOLM','ATTN','ER','STR_daily_ret','STR_turnover']]
    # res = daily_df[['DATE', 'TICKER', 'ILLIQ', 'Amivest', 'HL', 'HLI', 'ABNRETED','ABNRETM',
    #      'ABNVOLD', 'ABNVOLM','ATTN', 'ER', 'STR_daily_ret', 'STR_turnover']]
    return [res]

def help(daily_df,col,col_mkt,time_len=20):
    daily_df[f'sigma_{col}'] = np.abs(daily_df[col] - daily_df[col_mkt]) / (
                np.abs(daily_df[col]) + np.abs(daily_df[col_mkt] + 0.1))
    daily_df[f'sigma_{col}_rank'] = daily_df.groupby('TICKER')[f'sigma_{col}'].rolling(time_len, 5).rank(
        ascending=False).values  # ,method='first'
    daily_df[f'weight_{col}_help'] = pow(0.7, daily_df[f'sigma_{col}_rank']) * (1 / time_len)
    daily_df[f'weight_{col}_help'] = daily_df.groupby('TICKER')[f'weight_{col}_help'].rolling(time_len, 5).sum().values
    daily_df[f'weight_{col}'] = pow(0.7, daily_df[f'sigma_{col}_rank']) / daily_df[f'weight_{col}_help']
    daily_df[f'weight_{col}_roll20'] = daily_df.groupby('TICKER')[f'weight_{col}'].rolling(20, 5).mean().values
    daily_df[f'{col}_roll20'] = daily_df.groupby('TICKER')[col].rolling(20, 5).mean().values
    daily_df[f'weight_{col}_ret'] = daily_df[f'weight_{col}'] * daily_df[col]
    daily_df[f'weight_{col}_ret_roll20'] = daily_df.groupby('TICKER')[f'weight_{col}'].rolling(20, 5).mean().values
    daily_df[f'STR_{col}'] = daily_df[f'weight_{col}_ret_roll20'] - daily_df[f'weight_{col}_roll20'] * daily_df[f'{col}_roll20']
    return daily_df

def update(today):
    # todo: 这因子建议直接从头跑到尾，因为它其中有因子跨度在225
    res = run(start='20200101', end=today)
    for df in res:
        for col in df.columns[2:]:
            tmp = df[['DATE', 'TICKER', col]]
            print(tmp)
            feather.write_dataframe(tmp, os.path.join(DataPath.save_path_old, col + '.feather'))
            feather.write_dataframe(tmp, os.path.join(DataPath.save_path_update, col + '.feather'))

if __name__=='__main__':
    # path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\orderbook\20251029'
    # # cal('000001.feather')
    # run(start='20250101', end='20251029')
    update('20251030')