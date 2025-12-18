# (也可以diff主动买或主动卖，看情绪变化激增或骤减的情况，这些区间挑选出来)
# order buy和实际成交之间的关系
import os

import feather
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import read_min_data, get_tar_date, process_na_stock


def cal(date):
    df=read_min_data(date)#[['TICKER','min','active_buy_volume','active_sell_volume','open','close']]
    # print(df.columns)
    df=df[(df['min']!=930)&(df['min']<1457)]
    df.sort_values(['TICKER', 'min'], inplace=True)
    df['min_ret']=df['close']/df['open']-1
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    tmp1=df.groupby('TICKER')['min_ret'].std()#.reset_index().rename(columns={'min_ret':'risk'})
    tmp2=df.groupby('TICKER')['min_ret'].mean()
    tmp=tmp2+tmp1
    tmp=tmp.reset_index().rename(columns={'min_ret':'min_ret_risk'})
    tmp['DATE']=date
    tmp.columns=['TICKER','min_ret_risk','DATE']
    return tmp

def compute_time_decay_weights(m, H):
    j_values = np.arange(1, m + 1)
    numerator = 2 ** (-(m - j_values + 1) / H)
    denominator = np.sum(2 ** (-j_values / H))
    weights = numerator / denominator
    return weights

def weighted_dot(x):
    weight = compute_time_decay_weights(len(x), int(len(x)/2))
    return np.dot(x, weight)

def run(start,end):
    date_list=get_tar_date(start,end)
    # res=[]
    # for date in tqdm(date_list):
    #     tmp=cal(date)
    #     res.append(tmp)
    res=Parallel(n_jobs=13)(delayed(cal)(date) for date in tqdm(date_list))
    res=pd.concat(res).reset_index(drop=True)
    res.reset_index(drop=True,inplace=True)
    # feather.write_dataframe(res,r'C:\Users\admin\Desktop\test.feather')
    res.replace([np.inf,-np.inf],np.nan,inplace=True)
    # print(res)
    res=process_na_stock(res,'min_ret_risk')
    res.sort_values(['TICKER', 'DATE'], inplace=True)
    # t=21, 计算过去20天均值
    res['adjust']=res.groupby('TICKER')['min_ret_risk'].transform(lambda x: x.rolling(20, 5).mean())
    res['min_ret_risk_adjust']=res['adjust']-res['min_ret_risk']
    # res['risk_adjust'] = res.groupby('TICKER')['risk_adjust'].rolling(20, 5).apply(weighted_dot, raw=False).values
    # print(res)
    # --------------
    daily_df=feather.read_dataframe(DataPath.daily_path)
    daily_df.sort_values(['TICKER','DATE'],inplace=True)
    daily_df['daily_ret']=daily_df.groupby('TICKER')['close'].pct_change()
    daily_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    daily_df=daily_df[(daily_df['DATE']>=start)&(daily_df['DATE']<=end)]
    daily_df=daily_df[['DATE','TICKER','daily_ret']]
    # --------------
    # mkt_df=pd.read_csv(DataPath.wind_A_path)
    mkt_df=pd.read_csv(r'\\DESKTOP-79NUE61\Data_Storage2\881001.WI.csv')
    mkt_df['DATE']=mkt_df['DATE'].astype(str)
    mkt_df.sort_values(['DATE'],inplace=True)
    mkt_df['mkt_ret']=mkt_df['close'].pct_change()
    mkt_df=mkt_df[(mkt_df['DATE']>=start)&(mkt_df['DATE']<=end)][['DATE','mkt_ret']]
    # ---------------
    df=daily_df.merge(mkt_df,on='DATE',how='inner').merge(res,on=['DATE','TICKER'],how='inner')
    # df['risk_adjust']=1
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.dropna(inplace=True)
    df=process_na_stock(df,'daily_ret')
    df.sort_values(['TICKER','DATE'],inplace=True)
    df['min_ret_risk_factor']=df['min_ret_risk_adjust']*(df['daily_ret']-df['mkt_ret']) #np.abs()
    df['risk_1_factor']=(df['daily_ret']-df['mkt_ret']) #np.abs()
    df['min_ret_risk_factor_adjust'] = df.groupby('TICKER')['min_ret_risk_factor'].rolling(20, 5).apply(weighted_dot,
                                                            raw=False).reset_index(level=0,drop=True)
    df['risk_1_factor_adjust'] = df.groupby('TICKER')['risk_1_factor'].rolling(20, 5).apply(weighted_dot,
                                                            raw=False).reset_index(level=0,drop=True)
    df=df[['DATE','TICKER','min_ret_risk_factor_adjust','risk_1_factor_adjust']]
    # print(df)
    return [df]

def update_muli(filename,today,run,num=-50):
    if os.path.exists(os.path.join(DataPath.save_path_old,filename)) and False:
    # if False:
        print('因子更新中')
        old=feather.read_dataframe(os.path.join(DataPath.save_path_update,filename))
        new_start=sorted(old.DATE.drop_duplicates().to_list())[num]
        res=run(start=new_start,end=today)
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                # print(tmp)
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
        res=run(start='20200101',end=today)
        # res=run(start='20211201',end='20250801')
        # res=run(start='20250828',end='20250829')
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                print(tmp)
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))
                # feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))


def update(today):
    update_muli('min_ret_risk_factor_adjust.feather',today,run,-120)

if __name__=='__main__':
    # run('20250102','20250106')
    update('20251103')