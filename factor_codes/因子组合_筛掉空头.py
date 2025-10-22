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
    # file_list.remove('RSkew_daily.feather')
    # file_list.remove('SSF_v2_half10.feather')
    # file_list.remove('VaR_5%_v2.feather')
    # file_list = random.sample(file_list, 20)
#     file_list=['close_top24_ratio_roll20.feather','ctr.feather','expand_fell_roll20.feather','fear_idvi_fac.feather',
#                'net_midd_ratio.feather ','net_trade_ratio.feather','not_iso_num_imbalance_roll20_neutral_adjust.feather'
# ,'num_imbalance_roll20_neutral_adjust.feather','open_buy_will_ratio_roll20.feather','open_buy_will_str_roll20.feather',
#                'OvernightSmart20.feather','retOnNetAct_bottom_24_roll20.feather','Traction_LUD_amount.feather',
#                'Traction_LUD_trade_num.feather','tree_soldier_ret.feather','tree_soldier_vol.feather',
#     'turnTail4Sum_ratio.feather','vol_bottom24_ratio_roll20.feather ','有效100s卖出额_roll20.feather','随波逐流.feather']
    file_list=['close_top24_ratio_roll20.feather','net_midd_ratio.feather','net_trade_ratio.feather',
               'Traction_LUD_amount.feather','Traction_LUD_trade_num.feather','vol_bottom24_ratio_roll20.feather',
               '随波逐流.feather']
    bt_path=r'\\Desktop-79nue61\因子测试结果\田逸心\原因子增量回测20-21'
    start='20220101'
    end='20250901'
    pos_factors_list =[]
    neg_factors_list =[]
    res=pd.DataFrame()
    for file in tqdm(file_list):
        # file=file+'.feather'
        tmp=feather.read_dataframe(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file}')
        tmp=tmp[(tmp['DATE']>=start)&(tmp['DATE']<=end)].reset_index(drop=True)
        # tmp=tmp[tmp['DATE']=='20250102'].reset_index(drop=True)
        bt=pd.read_csv(os.path.join(bt_path,file.replace('feather','csv')))
        the_col=np.setdiff1d(tmp.columns,['DATE','TICKER'])
        tmp.dropna(inplace=True)
        tmp.sort_values(['DATE',the_col.tolist()[0]],inplace=True)
        tmp.reset_index(drop=True,inplace=True)
        if bt.tail(1).iloc[0,1]<0 and (bt.head(10)['分层收益']<0).any(): #负向，取小值
            tmp=tmp.groupby('DATE',as_index=False).apply(lambda x: x.head(int(len(x)*0.9))).reset_index(drop=True)
            neg_factors_list.append(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file}')
        elif bt.tail(1).iloc[0,1]>0 and (bt.head(10)['分层收益']<0).any():
            tmp = tmp.groupby('DATE',as_index=False).apply(lambda x: x.tail(int(len(x) * 0.9))).reset_index(drop=True)
            pos_factors_list.append(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file}')
        if res.empty:
            res=tmp[['DATE','TICKER']]
        else:
            res=res.merge(tmp[['DATE','TICKER']],on=['DATE','TICKER'],how='inner')
        print(res.shape)
    print('pos_factors_list =',pos_factors_list)
    print('neg_factors_list =',neg_factors_list)
    feather.write_dataframe(res,fr'C:\Users\admin\Desktop\codes.feather')

if __name__=='__main__':
    tar_codes()
    if os.path.exists(r'C:\Users\admin\Desktop\codes.feather'):
        df=feather.read_dataframe(r'C:\Users\admin\Desktop\codes.feather')
        df['DATE']=df['DATE'].astype(str)
    ret_df=feather.read_dataframe(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\return_因子组合用.feather')
    df=df.merge(ret_df,on=['DATE','TICKER'],how='left')
    res=df.groupby('DATE').agg({'TICKER':'size','return':'mean'}).reset_index()
    dd=['20240206', '20240207', '20240208','20240926', '20240927', '20240930', '20241008']
    res=res[~res['DATE'].isin(dd)]
    print(res)
    # feather.write_dataframe(res,r'C:\Users\admin\Desktop\res.feather')
    res.to_csv(r'C:\Users\admin\Desktop\res.csv')
