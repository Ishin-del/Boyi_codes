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

class DataPath():
    # 数据存储路径-----------------------------------------------------------------
    factor_out_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子'
    save_path_old=r'D:\tyx\检查更新\2020-2022.12'
    save_path_update=r'D:\tyx\检查更新\2021.6-2025.8'
    save_help_path=r'D:\tyx\raw_data'
    # tmp_path=r'D:\tyx\中间数据'
    # tmp_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\100s_sz_new'
    # tmp_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\100s_sz_new_min'
    tmp_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\100s_sz_new_max'

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
        # new_start='20220915'
        df1=run(start=new_start,end=today)
        for df in df1:
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
                    tt=test[~np.isclose(test.iloc[:, 2], test.iloc[:, 3])]
                    tt.columns=['DATE','TICKER','f1','f2']
                    print(tt)
                    print('检查更新出错!')
                    exit()
    else:
        print('因子生成')
        df1=run(start='20200101',end='20221221')
        # df1,df2=run(start='20240101',end='20250101')
        for df in df1:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                print(tmp)
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))
                # feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))

def get_tar_date(start,end):
    calender = pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'))
    calender['trade_date'] = calender['trade_date'].astype(str)
    calender = calender[(calender.trade_date >= start) & (calender.trade_date <= end)]
    tar_date = sorted(list(calender.trade_date.unique()))
    return tar_date

def read_min_data(date):
    warnings.filterwarnings('ignore')
    try:
        df_sz=feather.read_dataframe(os.path.join(DataPath.sz_min,date+'.feather'))
    except FileNotFoundError:
        df_sz=pd.DataFrame()
    try:
        df_sh=feather.read_dataframe(os.path.join(DataPath.sh_min,date+'.feather'))
    except FileNotFoundError:
        df_sh=pd.DataFrame()
    df=pd.concat([df_sh,df_sz]).reset_index(drop=True)
    # todo:
    df=df[(df['min']!=930)&(df['min']<1457)]
    # todo: 需要的前10%，20%， 最高最低
    df=df[['DATE','TICKER','min','volume','close']]
    tmp1 = df.groupby('TICKER')['volume'].mean()
    tmp1.replace([np.inf,-np.inf],np.nan,inplace=True)
    # -------------------------------------------------------------------
    #vol为基础
    # 最低10%
    df.sort_values(['TICKER','volume'],inplace=True) # todo:volume排序变成价格close
    res1 = help(df, 24, tmp1, 'vol_bottom24_ratio')
    # ----------------------------------------------------------------------------
    # close为基础
    df.sort_values(['TICKER', 'close'], ascending=[True, False], inplace=True)  # todo:volume排序变成价格close
    # 最高10%
    res2 = help(df, 24, tmp1, 'close_top24_ratio')
    res=res1.merge(res2,on='TICKER',how='inner')#.merge(res3,on='TICKER',how='inner').merge(res4,on='TICKER',how='inner')
    res['DATE']=date
    # return res[['DATE','TICKER','vol_top24_ratio', 'vol_top47_ratio', 'vol_bottom24_ratio','vol_bottom47_ratio']]
    # return res[['DATE','TICKER','close_top24_ratio', 'close_top47_ratio', 'close_bottom24_ratio','close_bottom47_ratio']]
    return res[['DATE','TICKER','close_top24_ratio', 'vol_bottom24_ratio']]

def help(df,top_num,tmp1,new_name):
    tmp = df.groupby('TICKER').head(top_num)  # 24,47
    tmp2 = tmp.groupby('TICKER')['volume'].mean()
    tmp2.replace([np.inf, -np.inf], np.nan, inplace=True)
    res = tmp2 / tmp1
    res = res.reset_index().rename(columns={'volume': new_name}) #'vol_top_ratio'
    return res

def run(start, end):
    tar_list=get_tar_date(start, end)
    # res=[]
    # for date in tqdm(tar_list):
    #     tmp=read_min_data(date)
    #     res.append(tmp)
    res=Parallel(n_jobs=12)(delayed(read_min_data)(date) for date in tqdm(tar_list))
    res=pd.concat(res).reset_index(drop=True)
    res.sort_values(['TICKER','DATE'],inplace=True)
    res.replace([np.inf, -np.inf], np.nan, inplace=True)
    res=process_na_stock(res,'vol_bottom24_ratio')
    res['vol_bottom24_ratio_roll20']=res.groupby('TICKER')['vol_bottom24_ratio'].rolling(20,5).mean().values
    res['close_top24_ratio_roll20'] = res.groupby('TICKER')['close_top24_ratio'].rolling(20, 5).mean().values
    res.replace([np.inf, -np.inf], np.nan, inplace=True)
    # res=res[['DATE','TICKER','close_top24_ratio_roll20','close_top47_ratio_roll20','close_bottom24_ratio_roll20','close_bottom47_ratio_roll20']]
    res=res[['DATE','TICKER','vol_bottom24_ratio_roll20','close_top24_ratio_roll20']]
    return [res]

def update(today='20251010'):
    update_muli('close_top24_ratio_roll20.feather',today,run)

if __name__=='__main__':
    # read_min_data('20250102')
    # run('20240109','20240110')
    # get_tick_data('20250102')
    update('20251010')
    # tar_list = get_tar_date('20200101', '20251009')
    # Parallel(n_jobs=2)(delayed(get_tick_data)(date) for date in tqdm(tar_list))
