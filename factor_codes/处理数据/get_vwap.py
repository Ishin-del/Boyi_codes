import warnings
from tqdm import tqdm
import feather
import os
import numpy as np
import pandas as pd
from joblib import delayed,Parallel


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

def cal(date,save_path):
    warnings.filterwarnings('ignore')
    try:
        df1=feather.read_dataframe(fr'\\Desktop-79nue61\SH\{date}_trade_sh.feather')
        df2=feather.read_dataframe(fr'\\Desktop-79nue61\SZ\{date}_trade_sz.feather')
    except FileNotFoundError:
        return
    ll=[]
    for df in [df1,df2]:
        df=df[['TickTime','Price','Qty','TICKER']]
        df['time']=df['TickTime']//100000
        df['amount']=df['Price']*df['Qty']
        df.replace([np.inf,-np.inf],np.nan,inplace=True)
        # -------
        time_list=[[930,1000],[930,1030],[930,1130],[1300,1500],[1400,1500],[1430,1500],[1000,1030]]
        res=[]
        for tl in time_list:
            tmp=df[(df['time']>=tl[0])&(df['time']<tl[1])]
            amn=tmp.groupby('TICKER')['amount'].sum()
            qty=tmp.groupby('TICKER')['Qty'].sum()
            r=amn/qty
            r.replace([np.inf,-np.inf],np.nan,inplace=True)
            if not r.isna().all():
                tt=df[df['TICKER'].isin(r[r.isna()].index)]
                tt_vwap=tt.groupby('TICKER')['amount'].sum()/tt.groupby('TICKER')['Qty'].sum()
                r=r.combine_first(tt_vwap)
                r.dropna(inplace=True)
            res.append(r)
        res=pd.concat(res,axis=1,join='inner').reset_index()
        res['DATE']=date
        res.columns=['TICKER','vwap_930_1000','vwap_930_1030','vwap_930_1130','vwap_1300_1500','vwap_1400_1500'
            ,'vwap_1430_1500','vwap_1000_1030','DATE']
        ll.append(res)
    res=pd.concat(ll).reset_index(drop=True)
    # print(res)
    tar_col=np.setdiff1d(res.columns,['TICKER','DATE'])
    os.makedirs(os.path.join(save_path,date),exist_ok=True)
    for col in tar_col:
        feather.write_dataframe(res,os.path.join(save_path,date,col+'.feather'))

def run():
    start=input('请输入开始时间：')
    end=input('请输入结束时间：')
    save_path=input('请输入数据存储路径：')
    tar_date=get_tar_date(start,end)
    Parallel(n_jobs=10)(delayed(cal)(date,save_path) for date in tqdm(tar_date,desc='数据计算中'))

if __name__=='__main__':
    # cal('20200102')
    run()