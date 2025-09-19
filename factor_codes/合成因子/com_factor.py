import itertools
import os
import warnings

import feather
import pandas as pd
from tqdm import tqdm
import numpy as np

from tool_tyx.tyx_funcs import orth


def get_factors():
    factors=os.listdir(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\中性化数据\拟使用量价因子')
    df=pd.DataFrame()
    for fac in tqdm(factors):
        tmp=feather.read_dataframe(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\中性化数据\拟使用量价因子\{fac}')
        if df.empty:
            df=tmp
        else:
            df=df.merge(tmp,on=['TICKER','DATE'],how='inner')
    corr_df = df.iloc[:, 2:].corr()
    corr_df=corr_df.where(np.abs(corr_df) < 0.6)
    feather.write_dataframe(df,r'C:\Users\admin\Desktop\factors.feather')
    feather.write_dataframe(corr_df,r'C:\Users\admin\Desktop\corr_df.feather')

def get_valid_fac():
    warnings.filterwarnings('ignore')
    full_df=feather.read_dataframe(r'C:\Users\admin\Desktop\factors.feather')#[['DATE','TICKER',
# 'AmtPerTrd_rolling20','ApT_outFlow_ratio_rolling20','GTR','OvernightSmart20','RSkew_daily_x','RSkew_daily_y','随波逐流']]
                 # 'DTGD','net_trade_ratio','sell_day_close_ratio','buy_concentration','entropy_adjust_std20']]
    # ll=os.listdir(r'\\Desktop-79nue61\因子测试结果\田逸心\20250919_中证2000_20220104_20241231_1')
    # dir=[]
    # for f in ll:
    #     if f in ['MSF_half10_processed','SSF_half10_processed','SSF_v2_half10_processed']:
    #         name=f.replace('half','')
    #     elif f in ['TGD_DTGD_processed','TGD_G_d_processed','TGD_G_u_processed','TGD_TGD_processed','TGD_time_diff_processed']:
    #         name=f[4:]
    #     else:
    #         name=f
    #     tmp=pd.read_excel(fr'\\Desktop-79nue61\因子测试结果\田逸心\20250919_中证2000_20220104_20241231_1\{f}\{name.replace('_processed','')}.xlsx')
    #     tt=[tmp['factor_name'].values[0],tmp['RANK_IC'].values[0]]
    #     print(tt)
    #     dir.append(tt)
    # dir_df=pd.DataFrame(dir)
    # dir_df.columns=['fac','ic']
    dir_df=feather.read_dataframe(r'C:\Users\admin\Desktop\dir_df.feather').T
    dir_df.columns = dir_df.loc['fac', :]
    dir_df=dir_df.iloc[1:, :]
    tt_ll=['AmtPerTrd_rolling20','big_buy_ratio','ideal_v_high','Mom_bigOrder_70_rolling20','num_imbalance_roll20_neutral_adjust',
           'open_buy_will_str_roll20','OvernightSmart20','sell_day_close_ratio','G_u','TGD','勇攀高峰_pure','随波逐流']
    # tt_ll=[]
    tar_list=list(itertools.combinations(tt_ll, 5))# todo:
    # 方向
    # df['buy_concentration']*=-1
    # df['sell_day_close_ratio']*=-1
    # df['ApT_outFlow_ratio_rolling20']*=-1
    # df['OvernightSmart20']*=-1
    # df['DTGD']*=-1
    # df['随波逐流']*=-1
    # z-score
    count=1
    for col_list in tqdm(tar_list):
        df=full_df[['DATE','TICKER']+list(col_list)]
        for col in df.columns[2:].tolist():
            # print(col)
            if col.startswith('RSkew_daily') or col.startswith('MSF_10'):
                df[col] = df.groupby('DATE')[col].apply(lambda x: (x - x.mean()) / x.std()).values
                continue
            elif dir_df.loc['ic',col]>0:
                df[col]*=-1
            df[col]=df.groupby('DATE')[col].apply(lambda x:(x-x.mean())/x.std()).values
        # 正交
        tmp=df.groupby('DATE')[df.columns[2:].tolist()].apply(orth).reset_index().drop(columns=['level_1'])
        tmp['TICKER']=df.groupby('DATE')['TICKER'].apply(lambda x:x).values
        # 相加
        tmp['com_factor']=tmp[df.columns[2:].tolist()].sum(axis=1)
        df=tmp[['DATE','TICKER','com_factor']]
        print(f'{count}:{list(col_list)}')
        feather.write_dataframe(df,fr'C:\Users\admin\Desktop\复合因子\com_fac{count}.feather')
        count += 1
if __name__=='__main__':
    get_valid_fac()