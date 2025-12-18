# @Author: Yixin Tian
# @File: trade_imbalance.py
# @Date: 2025/9/12 9:05
# @Software: PyCharm
import os
import warnings
import time
import feather
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
from tool_tyx.tyx_funcs import get_tar_date, process_na_stock, neutralize_factor


class DataPath():
    # 数据存储路径-----------------------------------------------------------------
    factor_out_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子'
    save_path_old=r'D:\tyx\检查更新\2020-2022.12'
    save_path_update=r'D:\tyx\检查更新\2021.6-2025.8'
    save_help_path=r'D:\tyx\raw_data'
    # tmp_path=r'D:\tyx\中间数据'
    tmp_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用'

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
    to_df_path=r'\\192.168.1.101\local_data\Data_Storage' #float_mv,adj_factors的数据路径,citic_code.feather数据,money_flow
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

def mark_isolated_trades(group):
    group = group.copy()
    group['is_isolated'] = True
    # 使用shift检查前后行的时间差
    group['prev_time_diff'] = (group['tmp_time'] - group['tmp_time'].shift(1)).abs()
    group['next_time_diff'] = (group['tmp_time'].shift(-1) - group['tmp_time']).abs()
    # 如果前后100ms内有交易，标记为非孤立
    group.loc[(group['prev_time_diff'] <= pd.Timedelta('100ms')) |
              (group['next_time_diff'] <= pd.Timedelta('100ms')), 'is_isolated'] = False
    return group.drop(['prev_time_diff', 'next_time_diff'], axis=1)


def get_tick_data(date):
    # 跑一个用了117s
    warnings.filterwarnings('ignore')
    if os.path.exists(os.path.join(DataPath.tmp_path, 'trade_imbalance', f'{date}.feather')):
        df = feather.read_dataframe(os.path.join(DataPath.tmp_path, 'trade_imbalance', f'{date}.feather'))
        return df
    try:
        sh_trade=feather.read_dataframe(os.path.join(DataPath.feather_sh,f'{date}_trade_sh.feather')).drop(
            columns=['Price','TradeMoney','TickBSFlag', 'ChannelNo'])
    except FileNotFoundError:
        sh_trade=pd.DataFrame()
    try:
        sz_trade=feather.read_dataframe(os.path.join(DataPath.feather_sz,f'{date}_trade_sz.feather')).drop(
            columns=['Price','TradeMoney','TickBSFlag', 'ChannelNo'])
    except FileNotFoundError:
        sz_trade=pd.DataFrame()
    df=pd.DataFrame()
    for tmp in [sh_trade,sz_trade]:
        tmp['TickBSFlag']=np.where(tmp['BuyOrderNo']>tmp['SellOrderNo'],'B','S')
        tmp.drop(columns=['BuyOrderNo', 'SellOrderNo','Qty'],inplace=True)
        # 识别孤立与否
        tmp['tmp_time'] = pd.to_datetime(tmp['TickTime'], format='%H%M%S%f')
        tmp = tmp.sort_values(['TICKER', 'tmp_time']).reset_index(drop=True)
        tmp = tmp.groupby('TICKER').apply(mark_isolated_trades).reset_index(drop=True)
        tmp.drop(columns=['tmp_time'],inplace=True)
        # 计算开始：
        tmp1=pd.DataFrame(tmp.groupby(['TICKER','TickBSFlag'])['TickIndex'].count()).rename(
            columns={'TickIndex':'trade_num'}).reset_index().pivot(index='TICKER',columns='TickBSFlag')
        tmp1.columns = ['buy_num', 'sell_num']
        tmp2=pd.DataFrame(tmp.groupby(['TICKER','TickBSFlag','is_isolated'])['TickIndex'].count()).reset_index().pivot(index='TICKER',columns=['TickBSFlag','is_isolated'])
        tmp2.columns = ['not_iso_buy_num', 'iso_buy_num', 'not_iso_sell_num', 'iso_sell_num']
        # tmp2=pd.DataFrame(tmp.groupby(['TICKER','TickBSFlag'])['Qty'].sum()).rename(columns={'Qty':'totoal_volume'})
        tmp=tmp1.merge(tmp2,left_index=True,right_index=True,how='inner').reset_index()
        # 部分因子值计算：
        tmp['num_imbalance'] = (tmp['buy_num'] - tmp['sell_num']) / (tmp['buy_num'] + tmp['sell_num'])
        # tmp['vol_imbalance'] = (tmp['buy_volume'] - tmp['sell_volume']) / (tmp['buy_volume'] + tmp['sell_volume'])
        tmp['iso_num_imbalance']=(tmp['iso_buy_num'] - tmp['iso_sell_num']) / (tmp['iso_buy_num'] + tmp['iso_sell_num'])
        tmp['not_iso_num_imbalance']=(tmp['not_iso_buy_num'] - tmp['not_iso_sell_num']) / (tmp['not_iso_buy_num'] + tmp['not_iso_sell_num'])
        df=pd.concat([df,tmp]).reset_index(drop=True)
    df['DATE']=date
    df=df[['DATE','TICKER','num_imbalance','iso_num_imbalance','not_iso_num_imbalance']]
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    os.makedirs(os.path.join(DataPath.tmp_path,'trade_imbalance'),exist_ok=True)
    feather.write_dataframe(df,os.path.join(DataPath.tmp_path,'trade_imbalance',f'{date}.feather'))
    # print(df)
    return df

def run(start,end):
    # print(start,end)
    warnings.filterwarnings('ignore')
    tar_date=get_tar_date(start,end)
    df=Parallel(n_jobs=3)(delayed(get_tick_data)(date) for date in tqdm(tar_date))
    # df=[]
    # for date in tqdm(tar_date):
    #     tmp=get_tick_data(date)
    #     df.append(tmp)
    df = pd.concat(df).reset_index(drop=True)
    # 拉取shares股本数据
    tmp = feather.read_dataframe(os.path.join(DataPath.to_df_path, 'float_mv.feather'))
    tmp=tmp[(tmp['DATE'] >= start) & (tmp['DATE'] <= end)].reset_index(drop=True)
    df=df.merge(tmp,on=['DATE','TICKER'],how='left')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # 横截面做市值中性化
    df.sort_values(['TICKER', 'DATE'], inplace=True)

    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df = process_na_stock(df, col='num_imbalance')
    for col in ['num_imbalance','iso_num_imbalance','not_iso_num_imbalance']:
        df[col + '_roll20'] = df.groupby('TICKER')[col].rolling(20, 5).mean().values
    # 删nan值
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df = process_na_stock(df, col='num_imbalance')
    df.dropna(inplace=True, how='any', axis=0)
    for col in ['num_imbalance','iso_num_imbalance','not_iso_num_imbalance']:
        df = neutralize_factor(df, col + '_roll20', 'float_mv')
    # 拉取每日close,计算ret
    # region 改1
    df.sort_values(['TICKER', 'DATE'], inplace=True)
    tmp = feather.read_dataframe(DataPath.daily_path)[['DATE', 'TICKER', 'close']]
    tmp['daily_ret'] = tmp.groupby('TICKER')['close'].pct_change()
    tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
    tmp=process_na_stock(tmp,'daily_ret')
    tmp['ret_roll20'] = tmp.groupby('TICKER')['daily_ret'].rolling(20, 5).sum().values
    # 在merge之前求ret_roll20，这样数据最少nan
    df = df.merge(tmp[['TICKER', 'DATE', 'ret_roll20']], on=['DATE', 'TICKER'], how='left')
    # 累计涨跌幅与因子做正交
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df=process_na_stock(df,'num_imbalance_roll20_neutral')
    df.dropna(inplace=True, axis=0, how='any')
    for col in ['num_imbalance_roll20_neutral', 'iso_num_imbalance_roll20_neutral','not_iso_num_imbalance_roll20_neutral']:
        # tmp = df[['DATE', 'TICKER', col, 'ret_roll20']]
        # df[col + '_adjust'] = tmp.groupby('DATE').apply(lambda x: orth(x[[col, 'ret_roll20']])).T.values
        df=neutralize_factor(df, col , 'ret_roll20',label='adjust')
    # endregion

    # # 拉取每日close,计算ret
    # df.sort_values(['TICKER', 'DATE'], inplace=True)
    # tmp = feather.read_dataframe(DataPath.daily_path)[['DATE', 'TICKER', 'close']]
    # df = df.merge(tmp, on=['DATE', 'TICKER'], how='left')
    # df['daily_ret'] = df.groupby('TICKER')['close'].pct_change()
    # df.replace([np.inf, -np.inf], np.nan, inplace=True)
    # df['ret_roll20']=df.groupby('TICKER')['daily_ret'].rolling(20,5).sum().values
    # # 累计涨跌幅与因子做正交
    # for col in ['num_imbalance_roll20_neutral','iso_num_imbalance_roll20_neutral','not_iso_num_imbalance_roll20_neutral']:
    #     tmp=df[['DATE','TICKER',col,'ret_roll20']]
    #     df[col+'_adjust']=tmp.groupby('DATE').apply(lambda x: orth(x[[col,'ret_roll20']])).T
    # df=df[['DATE', 'TICKER','num_imbalance_roll20','num_imbalance_roll20_neutral', 'iso_num_imbalance_roll20','iso_num_imbalance_roll20_neutral',
    #        'not_iso_num_imbalance_roll20','not_iso_num_imbalance_roll20_neutral','num_imbalance_roll20_neutral_adjust',
    #        'iso_num_imbalance_roll20_neutral_adjust','not_iso_num_imbalance_roll20_neutral_adjust']]
    df=df[['DATE', 'TICKER','num_imbalance_roll20_neutral_adjust','not_iso_num_imbalance_roll20_neutral_adjust']]
    # df=df[['DATE', 'TICKER','iso_num_imbalance_roll20_neutral_adjust','not_iso_num_imbalance_roll20_neutral_adjust']]
    # print(df)
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    return [df]

def update(today='20250929'):
    update_muli('not_iso_num_imbalance_roll20_neutral_adjust.feather',today,run,num=-120)

def update_muli(filename,today,run,num=-70):
    if os.path.exists(os.path.join(DataPath.save_path_update,filename)) and False: # and False
    # if False:
        print('因子更新中')
        old=feather.read_dataframe(os.path.join(DataPath.save_path_update,filename))
        # new_start=sorted(old.DATE.drop_duplicates().to_list())[num]
        # print(new_start)
        new_start='20221001'
        res=run(start=new_start,end=today)
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                # feather.write_dataframe(tmp, os.path.join(r'C:\Users\admin\Desktop', col + '.feather'))
                old=feather.read_dataframe(os.path.join(DataPath.save_path_update,col+'.feather'))
                test=old.merge(tmp,on=['DATE','TICKER'],how='inner').dropna()
                test.sort_values(['TICKER','DATE'],inplace=True)
                tar_list = sorted(list(test.DATE.unique()))[-5:]
                # print(tar_list)
                test = test[test.DATE.isin(tar_list)]
                if np.isclose(test.iloc[:,2],test.iloc[:,3]).all():
                    tmp=tmp[tmp.DATE>old.DATE.max()]
                    old=pd.concat([old,tmp]).reset_index(drop=True).drop_duplicates()
                    print(old)
                    feather.write_dataframe(old,os.path.join(DataPath.save_path_update,col+'.feather'))
                    feather.write_dataframe(old,os.path.join(DataPath.factor_out_path,col+'.feather'))
                else:
                    test.columns=['DATE','TICKER', 'f1','f2']
                    print(test[~np.isclose(test.iloc[:,2],test.iloc[:,3])])
                    # tt=test[~np.isclose(test.iloc[:, 2], test.iloc[:, 3])]
                    # feather.write_dataframe(tt,r'C:\Users\admin\Desktop\tt.feather')
                    print('检查更新出错!')
                    exit()
    else:
        print('因子生成中')
        # res=run(start='20200101',end='20221231')
        res=run(start='20200101',end=today)
        # res=run(start='20250828',end='20250829')
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                print(tmp)
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))
                # feather.write_dataframe(tmp, os.path.join(DataPath.factor_out_path, col + '.feather'))
                # feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))


if __name__=='__main__':
    # t1=time.time()
    # run('20250801','20250905')
    #
    # # get_tick_data('20250828')
    # t2=time.time()
    # print(t2-t1)
    update('20251023')