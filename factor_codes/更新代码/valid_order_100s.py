# @Author: Yixin Tian
# @File: valid_order_100s.py
# @Date: 2025/9/15 9:25
# @Software: PyCharm
import warnings
from joblib import Parallel,delayed
from tqdm import tqdm
import feather
import os
import pandas as pd
import numpy as np

from tool_tyx.tyx_funcs import process_na_stock


class DataPath():
    # 数据存储路径-----------------------------------------------------------------
    factor_out_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子'
    save_path_old=r'D:\tyx\检查更新\2020-2022.12'
    save_path_update=r'D:\tyx\检查更新\2021.6-2025.8'
    save_help_path=r'D:\tyx\raw_data'
    # tmp_path=r'D:\tyx\中间数据'
    tmp_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\100s_sz_new'

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

    order_min_sh=r'\\DESKTOP-79NUE61\SH_min_data_big_order'
    order_min_sz=r'\\DESKTOP-79NUE61\SZ_min_data_big_order'
    # -------------------------------------------
    # 机器学习数据路径
    train_data_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\rolling_generation_202508_sz_auction_end_20240815_908\train'
    train_big_order_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\rolling_generation_202508_sz_auction_end_20240815_909_big_order\train'

def update_muli(filename,today,run):
    if os.path.exists(os.path.join(DataPath.save_path_update,filename)):
        print('因子更新')
    # if False:
        old=feather.read_dataframe(os.path.join(DataPath.save_path_update,filename))
        new_start=sorted(old.DATE.drop_duplicates().to_list())[-90]
        df1,df2=run(start=new_start,end=today)
        for df in [df1,df2]:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                old=feather.read_dataframe(os.path.join(DataPath.save_path_update,col+'.feather'))
                test=old.merge(tmp,on=['DATE','TICKER'],how='inner').dropna().groupby('TICKER').tail(5)
                if np.isclose(test.iloc[:,2],test.iloc[:,3]).all():
                    tmp=tmp[tmp.DATE>old.DATE.max()]
                    old=pd.concat([old,tmp]).reset_index(drop=True).drop_duplicates()
                    print(old)
                    feather.write_dataframe(old,os.path.join(DataPath.save_path_update,col+'.feather'))
                    feather.write_dataframe(old,os.path.join(DataPath.factor_out_path,col+'.feather'))
                else:
                    print(test[~np.isclose(test.iloc[:,2],test.iloc[:,3])])
                    print('检查更新出错!')
                    exit()
    else:
        print('因子生成')
        df1,df2=run(start='20200101',end='20250929')
        for df in [df1,df2]:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                print(tmp)
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))

def get_tar_date(start,end):
    calender = pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'))
    calender['trade_date'] = calender['trade_date'].astype(str)
    calender = calender[(calender.trade_date >= start) & (calender.trade_date <= end)]
    tar_date = sorted(list(calender.trade_date.unique()))
    return tar_date

def order_tick_data(date,label='trade'):
    # print('1')
    try:
        sh_df=feather.read_dataframe(os.path.join(DataPath.feather_sh,f'{date}_{label}_sh.feather'))
    except FileNotFoundError:
        sh_df=pd.DataFrame()
    try:
        sz_df=feather.read_dataframe(os.path.join(DataPath.feather_sz,f'{date}_{label}_sz.feather'))
    except FileNotFoundError:
        sz_df=pd.DataFrame()
    return [sh_df,sz_df]


def get_tick_data(date):
    warnings.filterwarnings('ignore')
    if os.path.exists(os.path.join(DataPath.tmp_path,date)):
        # print('1')
        fac1=feather.read_dataframe(os.path.join(DataPath.tmp_path,date,'fac1.feather'))
        fac2=feather.read_dataframe(os.path.join(DataPath.tmp_path,date,'fac2.feather'))
        return fac1,fac2
    res_order=order_tick_data(date,'order')
    # print(res_order)
    fac1,fac2=pd.DataFrame(),pd.DataFrame()
    for tmp in res_order:
        if tmp.empty:
            continue
        # print(tmp)
        tmp['amount']=tmp['Price']*tmp['Qty']
        tmp = tmp[['TICKER', 'TickTime', 'Side', 'Type', 'amount']].reset_index(drop=True)
        tmp['TickTime'] //= 1000
        if tmp.TICKER.iloc[0][-2:]=='SZ':
            tmp['TickTime']=tmp['TickTime'] + int(date + '000000')
        tmp = tmp[(tmp.TickTime >= int(date + '100000')) & (tmp.TickTime < int(date + '145700'))]
        tmp['TickTime']=pd.to_datetime(tmp['TickTime'].astype(str), format='%Y%m%d%H%M%S')
        # 计算1s的和
        tmp.loc[tmp['Type'] == 'D', 'amount'] = -tmp['amount']
        record_df = tmp.groupby(['TICKER','TickTime', 'Side'])['amount'].sum().reset_index()
        tmp=record_df.groupby(['TICKER','TickTime'])['amount'].sum().reset_index()#.set_index('TickTime')
        # 选前100
        tmp.sort_values(['TICKER', 'amount'], ascending=[True, False], inplace=True)
        tmp=tmp.groupby('TICKER').head(100).reset_index(drop=True)
        # todo:
        # tmp = tmp.groupby('TICKER').tail(100).reset_index(drop=True)
        tmp=record_df.merge(tmp[['TICKER', 'TickTime']], on=['TICKER', 'TickTime'], how='inner')
        # 计算买入额 卖出额
        tt1=tmp[tmp.Side==1].groupby('TICKER')['amount'].sum().reset_index().rename(columns={'amount':'buy_amount'})
        tt2=tmp[tmp.Side==2].groupby('TICKER')['amount'].sum().reset_index().rename(columns={'amount':'sell_amount'})
        fac1=pd.concat([fac1,tt1]).reset_index(drop=True)
        fac2=pd.concat([fac2,tt2]).reset_index(drop=True)
    fac1['DATE']=date
    fac2['DATE']=date
    # print(fac1,fac2)
    fac1.replace([np.inf,-np.inf],np.nan,inplace=True)
    fac2.replace([np.inf,-np.inf],np.nan,inplace=True)
    os.makedirs(os.path.join(DataPath.tmp_path,date),exist_ok=True)
    feather.write_dataframe(fac1,os.path.join(DataPath.tmp_path,date,'fac1.feather'))
    feather.write_dataframe(fac2,os.path.join(DataPath.tmp_path,date,'fac2.feather'))
    return fac1,fac2

def run(start, end):
    tar_list=get_tar_date(start, end)
    # res=[]
    # for date in tqdm(tar_list):
    #     tmp=get_tick_data(date)
    #     res.append(tmp)
    res=Parallel(n_jobs=13)(delayed(get_tick_data)(date) for date in tqdm(tar_list))
    # res=pd.concat(res).reset_index(drop=True)
    res1=pd.concat([x[0] for x in res]).reset_index(drop=True)
    res2=pd.concat([x[1] for x in res]).reset_index(drop=True)
    res1.sort_values(['TICKER','DATE'],inplace=True)
    res2.sort_values(['TICKER','DATE'],inplace=True)
    res1.replace([np.inf,-np.inf],np.nan,inplace=True)
    res2.replace([np.inf,-np.inf],np.nan,inplace=True)
    res1=process_na_stock(res1,'buy_amount')
    res2=process_na_stock(res2,'sell_amount')
    res1['buy_amount_roll20']=res1.groupby('TICKER')['buy_amount'].rolling(20,5).mean().values
    res2['sell_amount_roll20']=res2.groupby('TICKER')['sell_amount'].rolling(20,5).mean().values
    res1=res1[['DATE','TICKER','buy_amount','buy_amount_roll20']]
    res2=res2[['DATE','TICKER','sell_amount','sell_amount_roll20']]
    res1.rename(columns={'buy_amount':'有效100s买入额','buy_amount_roll20':'有效100s买入额_roll20'},inplace=True)
    res2.rename(columns={'sell_amount':'有效100s卖出额','sell_amount_roll20':'有效100s卖出额_roll20'},inplace=True)
    # print(res1)
    # print(res2)
    return [res1,res2]

def update(today='20251009'):
    update_muli('有效100s买入额.feather',today,run)

if __name__=='__main__':
    # run('20240109','20240110')
    # get_tick_data('20250102')
    update()