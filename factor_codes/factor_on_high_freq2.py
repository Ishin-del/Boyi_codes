import os
import warnings
import feather
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from tqdm import tqdm

# 中信建投-因子深度研究系列：高频订单失衡及价差因子
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
    to_df_path=r'\\192.168.1.101\local_data\Data_Storage' #float_mv.feather,adj_factors的数据路径,citic_code.feather数据(行业数据),money_flow
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


    snap_path_2022=fr'\\192.168.1.7\data01\data_feather'
    snap_path_2023=fr'\\192.168.1.7\data02\data_feather'
    snap_path_2024=fr'\\192.168.1.7\data03\data_feather'
    snap_path_2025=fr'\\192.168.1.7\data04\data_feather'


def get_tar_date(start,end):
    calender = pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'))
    calender['trade_date'] = calender['trade_date'].astype(str)
    calender = calender[(calender.trade_date >= start) & (calender.trade_date <= end)]
    tar_date = sorted(list(calender.trade_date.unique()))
    return tar_date

def read_snap_data(code,path='',date='',snap_time=3):
    warnings.filterwarnings('ignore')
    # path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\orderbook\20251029'
    tmp=feather.read_dataframe(os.path.join(path,code))
    tmp.sort_values('TickTime',inplace=True)
    tmp['DateTime']=tmp['TickTime']//1000
    tmp=tmp[(tmp['DateTime']>=int(date+'093000'))&(tmp['DateTime']<int(date+'145800'))].reset_index(drop=True)
    tmp.drop_duplicates(subset='DateTime', keep='last',inplace=True)
    tmp=tmp.reset_index(drop=True).iloc[::snap_time]
    tmp.drop(columns=['TickName','ChannelNo','LastPrice','LastSeq','TickTime'],inplace=True)
    # 计算开始 ----------------------------------
    df=tmp.copy()
    df.sort_values(['DateTime'], inplace=True)
    # 因子一----------------------------
    w = np.array([1 - (i - 1) / 20 for i in range(1, 21)])
    tar_cols1 = [f'Bid_Qty_{i}' for i in range(20)]
    tar_cols2 = [f'Offer_Qty_{i}' for i in range(20)]
    df[f'VwB']= (df[tar_cols1] * w).sum(axis=1)/sum(w)
    df[f'VwA']= (df[tar_cols2] * w).sum(axis=1)/sum(w)
    for col in df.columns[2:]:
        # df[col + '_help'] = df.groupby('SecurityID')[col].shift(1)
        df[col + '_help'] = df[col].shift(1)
    for i in range(20):
        df[f'VB_{i}'] = np.where(df[f'Bid_{i}'] < df[f'Bid_{i}_help'], 0,
                                 np.where(df[f'Bid_{i}'] > df[f'Bid_{i}_help'], df[f'VwB'],
                                          df[f'VwB_help'] - df[f'VwB']))
        df[f'VA_{i}'] = np.where(df[f'Offer_{i}'] < df[f'Offer_{i}_help'], 0,
                                 np.where(df[f'Offer_{i}'] > df[f'Offer_{i}_help'], df[f'VwA'],
                                          df[f'VwA_help'] - df[f'VwA']))
        df[f'SOIR_{i}'] = (df[f'VB_{i}'] - df[f'VA_{i}']) / (df[f'VB_{i}'] + df[f'VA_{i}'])
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    tar_cols3=[f'SOIR_{i}' for i in range(20)]
    df['SOIR'] = (df[tar_cols3]*w).sum(axis=1)/sum(w)
    #高频转低频
    df['SOIR']=(df['SOIR']-df['SOIR'].mean())/df['SOIR'].std()
    tmp1=df['SOIR'].mean()
    # 因子二-----------------------------
    df['M_t'] = (df['Bid_0'] + df['Offer_0']) / 2
    for i in range(1, 6):
        df[f'M_k_{i}'] = df['M_t'].shift(i)
        df[f'MPC_{i}'] = df['M_t'] / df[f'M_k_{i}'] - 1
    df['MPC_max_t'] = df[['MPC_1','MPC_2','MPC_3','MPC_4','MPC_5']].max(axis=1)
    tmp2 = df['MPC_max_t'].max()
    tmp3 = df['MPC_max_t'].skew()
    # 因子三----------------------------
    # 这是第二篇研报
    for i in range(20):
        df[f'V2B_{i}'] = np.where(df[f'Bid_{i}'] < df[f'Bid_{i}_help'], -df[f'VwB'],
                                  np.where(df[f'Bid_{i}'] > df[f'Bid_{i}_help'], df[f'VwB'],
                                           df[f'VwB_help'] - df[f'VwB']))
        df[f'V2A_{i}'] = np.where(df[f'Offer_{i}'] < df[f'Offer_{i}_help'], -df[f'VwA'],
                                  np.where(df[f'Offer_{i}'] > df[f'Offer_{i}_help'], df[f'VwA'],
                                           df[f'VwA_help'] - df[f'VwA']))
        df[f'OFI_{i}'] = df[f'V2B_{i}'] - df[f'V2A_{i}']
    tar_cols4=[f'OFI_{i}' for i in range(20)]
    df['MOFI'] = df[tar_cols4].sum(axis=1)
    w2 = np.array([i/5 for i in range(1, 21)])
    df['MOFI_weight'] = (df[tar_cols4]*w2).sum(axis=1)/sum(w2)
    df['LogQuoteSlope'] = (np.log(df['Offer_0']) - np.log(df['Bid_0'])) / (np.log(df['Offer_0']) + np.log(df['Bid_0']))
    # 高频转低频
    tmp4=df['MOFI'].mean()
    tmp5=df['MOFI_weight'].mean()
    tmp6=df['LogQuoteSlope'].mean()
    tmp=pd.DataFrame([{'DATE':date,'TICKER':code.replace('.feather','.SZ'),'SOIR':tmp1,'MPC_max':tmp2,'MPC_skew':tmp3,
                      'MOFI':tmp4,'MOFI_weight':tmp5,'LogQuoteSlope':tmp6}])
    # print(tmp)
    return tmp

def cal(date,path='',code_files=[]):
    warnings.filterwarnings('ignore')
    if os.path.exists(fr'D:\tyx\中间数据\订单不平衡中间数据\{date}.feather'):
        tmp = feather.read_dataframe(fr'D:\tyx\中间数据\订单不平衡OIR\{date}.feather')
        return tmp
    if date[:4] == '2022':
        path = os.path.join(DataPath.feather_2022, date, 'Snapshot')
        code_files = os.listdir(path)
    elif date[:4] == '2023':
        path = os.path.join(DataPath.feather_2023, date, 'Snapshot')
        code_files = os.listdir(path)
    elif date[:4] == '2024':
        path = os.path.join(DataPath.feather_2024, date, 'Snapshot')
        code_files = os.listdir(path)
    elif date[:4] == '2025':
        path = os.path.join(DataPath.feather_2025, date, 'Snapshot')
        code_files = os.listdir(path)
    # code_files=os.listdir(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\orderbook\20251029')
    # for code in code_files:
    #     df=read_snap_data(code)
    df = Parallel(n_jobs=15)(delayed(read_snap_data)(code, path, date) for code in tqdm(code_files,desc=f'{date}数据计算中'))
    df = pd.concat(df).reset_index(drop=True)
    df['MOFI']=(df['MOFI']-df['MOFI'].mean())/df['MOFI'].std()
    df['MOFI_weight']=(df['MOFI_weight']-df['MOFI_weight'].mean())/df['MOFI_weight'].std()
    df['LogQuoteSlope']=(df['LogQuoteSlope']-df['LogQuoteSlope'].mean())/df['LogQuoteSlope'].std()
    print(df)
    os.makedirs(r'D:\tyx\中间数据\订单不平衡中间数据')
    feather.read_dataframe(df,fr'D:\tyx\中间数据\订单不平衡中间数据\{date}.feather')
    return df

def run(start,end):
    tar_date=get_tar_date(start,end)
    res=[]
    for date in tar_date:
        tmp=cal(date)
        res.append(tmp)
    res=pd.concat(res).reset_index(drop=True)
    return [res]

def update(today='20251030'):
    update_muli('LogQuoteSlope.feather',today,run)

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
                    # feather.write_dataframe(old,os.path.join(DataPath.factor_out_path,col+'.feather'))
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
    # cal('20251029')
    # read_snap_data('000001.feather')
    update()