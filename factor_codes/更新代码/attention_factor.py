import feather
import pandas as pd
import numpy as np
from joblib import Parallel,delayed
import os
import warnings
from tqdm import tqdm
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import process_na_stock

"""
中信建投：投资者有限关注及注意力捕捉与溢出
"""

def cal_factors1(df):
    warnings.filterwarnings('ignore')
    df=df.groupby('citic_code', as_index=False, group_keys=False).apply(cal_factors2)
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    return df

def cal_factors2(group):
    group.sort_values(by='float_mv', inplace=True)
    group.replace([np.inf,-np.inf],np.nan,inplace=True)
    tmp1=group.iloc[:int(len(group) * 0.3)]
    tmp1['att_overflow']=tmp1['attn']-tmp1['attn'].mean()

    tmp2=group.iloc[int(len(group) * 0.3):int(len(group) * 0.7)]
    tmp2['att_overflow']=tmp2['attn']-tmp2['attn'].mean()

    tmp3=group.iloc[int(len(group) * 0.7):len(group)]
    tmp3['att_overflow']=tmp3['attn']-tmp3['attn'].mean()

    tmp=pd.concat([tmp1,tmp2,tmp3])
    tmp.replace([np.inf,-np.inf],np.nan,inplace=True)
    return tmp


def run(start,end):
    warnings.filterwarnings('ignore')
    ind_data=feather.read_dataframe(DataPath.ind_df_path)
    ind_data=ind_data[(ind_data.DATE>=start)&(ind_data.DATE<=end)]

    daily_df=feather.read_dataframe(DataPath.daily_path,columns=['DATE','TICKER','close'])
    daily_df=daily_df[(daily_df.DATE>=start)&(daily_df.DATE<=end)]

    daily_df.sort_values(['TICKER','DATE'],inplace=True)
    daily_df['rtn']=daily_df.groupby('TICKER')['close'].pct_change()
    daily_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    daily_df['rtn_mean']=daily_df.groupby('DATE')['rtn'].transform('mean')
    daily_df['ab_norm_ret']=np.square(daily_df['rtn']-daily_df['rtn_mean'])
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    daily_df=process_na_stock(daily_df,'ab_norm_ret')
    daily_df['attn']=daily_df.groupby('TICKER')['ab_norm_ret'].rolling(21,5).mean().values  #.dropna(how='all')
    daily_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    daily_df.dropna(axis=0,how='any',inplace=True)

    market_cap=feather.read_dataframe(os.path.join(DataPath.to_df_path,'float_mv.feather'))
    market_cap=market_cap[(market_cap.DATE>=start) & (market_cap.DATE<=end)]
    daily_df=daily_df.merge(ind_data,on=['TICKER','DATE'],how='inner').merge(market_cap,on=['TICKER','DATE'],how='inner')
    # print(daily_df)
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    daily_df=process_na_stock(daily_df,'attn')
    daily_df.rename(columns={'CITIC_CODE':'citic_code'},inplace=True)
    daily_df=daily_df[['DATE','TICKER','attn','citic_code', 'float_mv']]
    tar_date_list=sorted(daily_df.DATE.unique().tolist())
    # print(tar_date_list)
    res= Parallel(n_jobs=15)(delayed(cal_factors1)(daily_df[daily_df.DATE==date]) for date in tqdm(tar_date_list))
    # for date in tqdm(tar_date_list):
    #     # print(date)
    #     # print(daily_df[daily_df.DATE==date])
    #     res=cal_factors1(daily_df[daily_df.DATE==date])
    #     print('-----------------')
    #     print(res)
    #     break
    # print(res)
    res=pd.concat(res).reset_index(drop=True)[['DATE','TICKER','att_overflow']]
    return [res]

def update(today='20251021'):
    update_muli('att_overflow.feather',today,run,num=-120)

def update_muli(filename,today,run,num=-50):
    if os.path.exists(os.path.join(DataPath.save_path_update,filename)) and False: # and False
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
                    feather.write_dataframe(old,os.path.join(DataPath.factor_out_path,col+'.feather'))
                else:
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
                feather.write_dataframe(tmp, os.path.join(DataPath.factor_out_path, col + '.feather'))
                # feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))

if  __name__=='__main__':
    update('20251023')