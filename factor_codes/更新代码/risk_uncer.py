# @Author: Yixin Tian
# @File: risk_uncer.py
# @Date: 2025/9/24 10:22
# @Software: PyCharm

"""
东北证券：基于高频数据的风险不确定性因子
"""
import feather
import numpy as np
import warnings
import statsmodels.api as sm
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import read_min_data, get_tar_date,process_na_stock
import os

def update_muli(filename,today,run,num=-100):
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
                tar_list = sorted(list(test.DATE.unique()))[-3:]
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

def linear_fun(df, y, x):
    X = sm.add_constant(df[x])
    y = df[y]
    model = sm.WLS(y, X).fit()
    df['resid']=model.resid
    return df

def get_data(date,mkt_df):
    warnings.filterwarnings('ignore')
    df=read_min_data(date)
    if df.empty:
        return
    df=df[(df['min']!=930)&(df['min']<1457)][['TICKER','DATE','close']]
    df.sort_values(['TICKER','DATE'],inplace=True)
    # df['ret']=df.groupby('TICKER')['close'].pct_change()
    df['ret']=df.groupby('TICKER')['close'].pct_change(periods=4)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df['log_ret']=np.log(1+df['ret'])
    df['log_ret_square']=df['log_ret']**2
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    tmp=df.groupby('TICKER')['log_ret_square'].sum().reset_index().rename(columns={'log_ret_square':'RV'})
    # -----------------------------------------------
    # 分位数选0.1
    df['VaR']=df.groupby('TICKER')['ret'].transform(lambda x:x.quantile(0.95))
    df['VaR_RT']=df.groupby('TICKER')['ret'].transform(lambda x:x.quantile(1-0.95))
    tmp1=-df[df['ret']<=df['VaR']].groupby('TICKER')['ret'].mean()#
    tmp1=tmp1.reset_index().rename(columns={'ret':'cVaR'})
    tmp2=df[df['ret']>=df['VaR_RT']].groupby('TICKER')['ret'].mean()
    tmp2=tmp2.reset_index().rename(columns={'ret':'cVaR_RT'})
    tmp=tmp.merge(tmp1,on='TICKER',how='inner').merge(tmp2,on='TICKER',how='inner')
    # -----------------------------------------------
    mkt_df['DATE']=mkt_df['DATE'].astype(str)
    df=df.merge(mkt_df,on='DATE',how='left')
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.dropna(subset=['mkt_ret','ret'],inplace=True)
    df['resid']=np.nan
    df=df.groupby('TICKER').apply(lambda x:linear_fun(x,'mkt_ret','ret')).reset_index(drop=True)
    # df=linear_fun(df,'mkt_ret','ret')
    df['resid_square']=df['resid']**2
    tmp1=df.groupby('TICKER')['resid_square'].sum().reset_index().rename(columns={'resid_square':'ID_RV'})
    tmp=tmp.merge(tmp1,on='TICKER',how='inner')
    tmp['DATE']=date
    # print(tmp)
    return tmp[['DATE','TICKER','RV','cVaR', 'cVaR_RT']] #, 'ID_RV'

def help_fun(res,col,out_col,window=20):
    res.sort_values(['TICKER', 'DATE'], inplace=True)
    res[col+'_base'] = res.groupby('TICKER')[col].shift(window)
    res[col+'_diff'] = res[col] - res[col+'_base']
    res[col+'_hat'] = res.groupby('TICKER')[col+'_diff'].transform(lambda x: x.rolling(window,int(window / 4)).sum())
    # 可能因为前面的值太小，股票300201从20220901到20221129都是nan值，导致数据对不上，用tail(3)对比没问题
    res[col+'_hat'] /= (window+1)
    res.replace([np.inf,-np.inf],np.nan,inplace=True)
    res[col+'_diff2'] = (res[col+'_diff'] - res[col+'_hat']) ** 2
    res[out_col + '_tmp'] = res.groupby('TICKER')[col + '_diff2'].transform(lambda x: x.rolling(window,int(window / 4)).mean())
    res.replace([np.inf, -np.inf], np.nan, inplace=True)
    res[out_col] = res[out_col+'_tmp'] / res[col+'_hat']
    res.replace([np.inf, -np.inf], np.nan, inplace=True)
    return res

def run(start,end):
    tar_list=get_tar_date(start,end)
    mkt_df = pd.read_csv(DataPath.wind_A_path)[['DATE', 'close']]
    mkt_df.sort_values('DATE', inplace=True)
    mkt_df['mkt_ret'] = mkt_df['close'].pct_change()
    mkt_df = mkt_df[(mkt_df.DATE >= int(start)) & (mkt_df.DATE <= int(end))]
    # res=[]
    # for date in tqdm(tar_list):
    #     tmp=get_data(date, mkt_df=mkt_df[mkt_df.DATE == int(date)][['DATE', 'mkt_ret']])
    #     res.append(tmp)
    #     print(tmp)
    res=Parallel(n_jobs=15)(delayed(get_data)(date,mkt_df[mkt_df.DATE == int(date)][['DATE', 'mkt_ret']]) for date in tqdm(tar_list))
    res=pd.concat(res).reset_index(drop=True)
    res=process_na_stock(res,'RV')
    res.replace([np.inf,-np.inf],np.nan,inplace=True)
    res['RV_roll5']=res.groupby('TICKER')['RV'].rolling(5,3).mean().values
    # res['ID_RV_roll5']=res.groupby('TICKER')['ID_RV'].rolling(5,3).mean().values
    # res['VaR_roll5']=res.groupby('TICKER')['VaR'].rolling(5,3).mean().values
    # res['VaR_TR_roll5']=res.groupby('TICKER')['VaR_TR'].rolling(5,3).mean().values
    res['cVaR_roll5']=res.groupby('TICKER')['cVaR'].rolling(5,3).mean().values
    res['cVaR_RT_roll5']=res.groupby('TICKER')['cVaR_RT'].rolling(5,3).mean().values
    # res=help_fun(res,'RV','VOV')
    # res=help_fun(res,'cVaR','VOTR_cVaR')
    # res=help_fun(res,'cVaR_RT','VOTR_cVaR_RT')
    # res=help_fun(res,'ID_RV','ID_VOV')
    # feather.write_dataframe(res, r'C:\Users\admin\Desktop\check1.feather')
    # res=res[['DATE','TICKER','RV_roll5','VOV','cVaR_roll5','cVaR_RT_roll5','VOTR_cVaR','VOTR_cVaR_RT','ID_RV_roll5','ID_VOV']]
    res=res[['DATE','TICKER','RV_roll5','cVaR_roll5','cVaR_RT_roll5']]
    # feather.write_dataframe(res, r'C:\Users\admin\Desktop\check1.feather')
    #'VaR_roll5','VaR_TR_roll5',
    return [res]
    # return res

def update(today='20250905'):
    update_muli('RV_roll5.feather',today,run)

if __name__=='__main__':
    # run('20250102','20250112')
    update()
    # res1=run('20220601','20221231')
    # feather.write_dataframe(res1, r'C:\Users\admin\Desktop\check1.feather')
    # res2=run('20220901','20230231')
    # feather.write_dataframe(res2, r'C:\Users\admin\Desktop\check2.feather')