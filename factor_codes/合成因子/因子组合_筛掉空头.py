import os
import random
import warnings
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm

def tar_codes():
    warnings.filterwarnings('ignore')
    # file_list=os.listdir(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子')
    file_list=['net_midd_ratio','net_trade_ratio']
   #  file_list=['net_midd_ratio','net_trade_ratio','TGD_G_d','vol_bottom24_ratio_roll20','open_buy_will_str_roll20'
   # , '随波逐流','AmtPerTrd_rolling20','fear_idvi_fac','ideal_v_high','tree_soldier_ret','tree_soldier_vol',
   #  '有效100s卖出额_roll20']

    # file_list=['net_trade_ratio', 'net_midd_ratio', 'TGD_G_d', 'open_buy_will_str_roll20', 'ideal_v_high', '随波逐流']
    # file_list.remove('VaR_5%_v2.feather')
    bt_path=r'\\Desktop-79nue61\因子测试结果\田逸心\原因子增量回测22-24'
    # start='20220101'
    # end='20250901'
    # end='20241231'
    pos_factors_list =[]
    neg_factors_list =[]
    res=pd.DataFrame()
    for file in tqdm(file_list):
        file=file+'.feather'
        tmp=feather.read_dataframe(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file}')
        # tmp=tmp[(tmp['DATE']>='20230102')&(tmp['DATE']<='20250930')]
        tmp.sort_values(['TICKER','DATE'],inplace=True)
        # tmp['DATE']=tmp.groupby('TICKER')['DATE'].shift(-1)
        bt=pd.read_csv(os.path.join(bt_path,file.replace('feather','csv')))
        the_col=list(np.setdiff1d(tmp.columns,['DATE','TICKER']))[0]
        tmp[the_col]=tmp.groupby('TICKER')[the_col].shift(1).reset_index(drop=True)
        # tmp = tmp[(tmp['DATE'] >= '20230102') & (tmp['DATE'] <= '20250930')]
        tmp.dropna(inplace=True)
        tmp.sort_values(['DATE',the_col],inplace=True)
        tmp.reset_index(drop=True,inplace=True)
        if bt.tail(1).iloc[0,1]<0 and (bt.head(10)['分层收益']<0).any(): #负向，取小值
            tmp=tmp.groupby('DATE',as_index=False).apply(lambda x: x.head(int(len(x)*0.9))).reset_index(drop=True)
            # tmp=tmp.groupby('DATE',as_index=False).apply(lambda x: x.head(int(len(x)*0.3))).reset_index(drop=True)
            neg_factors_list.append(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file}')
        elif bt.tail(1).iloc[0,1]>0 and (bt.head(10)['分层收益']<0).any():
            tmp = tmp.groupby('DATE',as_index=False).apply(lambda x: x.tail(int(len(x) * 0.9))).reset_index(drop=True)
            # tmp = tmp.groupby('DATE',as_index=False).apply(lambda x: x.tail(int(len(x) * 0.3))).reset_index(drop=True)
            pos_factors_list.append(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file}')
        if res.empty:
            res=tmp[['DATE','TICKER']]
        else:
            res=res.merge(tmp,on=['DATE','TICKER'],how='inner')
        # print(res.shape)
    print('pos_factors_list =',pos_factors_list)
    print('neg_factors_list =',neg_factors_list)
    # feather.write_dataframe(res,fr'C:\Users\admin\Desktop\codes.feather')
    return res
if __name__=='__main__':
    if os.path.exists(r'C:\Users\admin\Desktop\codes.feather'):
        df=feather.read_dataframe(r'C:\Users\admin\Desktop\codes.feather')
    else:
        df=tar_codes()
    df['DATE'] = df['DATE'].astype(str)
    # ret_df=feather.read_dataframe(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\因子组合收益return.feather')
    # ret_df=pd.read_csv(r'C:\Users\admin\Desktop\tickerpool1.csv',index_col=0).rename(columns={'ret':'return'})
    ret_df=pd.read_csv(r'C:\Users\admin\Desktop\tickerpool2.csv',index_col=0).rename(columns={'ret':'return'})
    ret_df.drop_duplicates(inplace=True)
    ret_df['DATE']=ret_df['DATE'].astype(str)
    df=df.merge(ret_df,on=['DATE','TICKER'],how='inner')
    df.sort_values(['TICKER','DATE'],inplace=True)
    df = df.dropna().reset_index(drop=True)
    res=df.copy()
    print(res.groupby('DATE').agg({'TICKER':'size'}))
    # print(res)
    dd=['20240206', '20240207', '20240208','20240926', '20240927', '20240930', '20241008']
    res=res[~res['DATE'].isin(dd)]
    res1=res[(res['DATE']>='20230101')&(res['DATE']<='20241231')]
    print('23年-24年均值:', res1['return'].mean())
    print('23年-24年中位数:', res1['return'].median())
    print('23年-24年胜率:', sum(res1['return'] > 0) / len(res))

    res = res[res['DATE'] >= '20250101']
    print('25年均值:',res['return'].mean())
    print('25年中位数:',res['return'].median())
    print('25年胜率:', res[res['return'] > 0].shape[0] / res.shape[0])

