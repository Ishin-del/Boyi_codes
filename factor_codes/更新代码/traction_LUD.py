# @Author: Yixin Tian
# @File: traction_LUD.py
# @Date: 2025/9/28 14:53
# @Software: PyCharm

"""
开源证券：从涨跌停外溢行为到股票关联网络
"""

import os
import feather
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import read_min_data, get_tar_date,process_na_stock
import statsmodels.api as sm
import warnings
def get_data(date):
    df=read_min_data(date)
    if df.empty:
        return
    # df=df[(df['min']!=930)&(df['min']<1457)]
    # df=df[(df['min']!=930)&(df['min']<=1000)]
    df=df[df['min']<=1000]
    df.sort_values(['TICKER','DATE'],inplace=True)
    df['min_ret']=df.groupby('TICKER')['close'].pct_change()
    return df[['DATE','TICKER','min','min_ret','close']]


def cal(date,ret_df):
    warnings.filterwarnings('ignore')
    df=get_data(date)
    df=df.merge(ret_df,on='TICKER',how='inner')
    df['dir_up']=np.where(df['min_ret']>0,1,0)
    df['dir_down']=np.where(df['min_ret']<0,1,0)
    useA=df[['TICKER','min','dir_up']].pivot(index='min',columns='TICKER',values='dir_up')
    useB=df[['TICKER','min','dir_down']].pivot(index='min',columns='TICKER',values='dir_down')
    useA=useA[useA.index != 930]
    useB=useB[useB.index != 930]
    # way2----------------------------------------------------
    up_res = np.dot(useA.T.values, useA.values)
    down_res = np.dot(useB.T.values, useB.values)
    # result_up = (len(useA)+up_res) / 2 / len(useA)
    # result_down = (len(useA)+down_res)/ 2 / len(useA)
    result_up = up_res / len(useA)
    result_down =down_res/ len(useB)
    result_df1 = pd.DataFrame(result_up, index=useA.T.index, columns=useA.columns).T
    result_df2 = pd.DataFrame(result_down, index=useB.T.index, columns=useB.columns).T
    result_df=result_df1+result_df2
    # --------------------------------------
    num=int(len(result_df)/2)
    tmp_values=result_df.apply(lambda x: x.nlargest(num).values)
    tmp_codes=result_df.apply(lambda x: x.nlargest(num).index)
    tmp_dict = ret_df.set_index('TICKER')['ret_roll20'].to_dict()
    tmp_ret=tmp_codes.applymap(lambda x: tmp_dict.get(x, x))
    res=(tmp_values*tmp_ret).sum()/tmp_values.sum()
    res=pd.DataFrame(res).reset_index()
    res['DATE']=date
    res=res[['DATE','TICKER',0]].rename(columns={0:'exp_LUD'})
    return res


def neutralize_factor(df, factor_col, mv_col, industry_col):
    data = df.copy()
    industry_dummies = pd.get_dummies(data[industry_col], prefix='industry',dtype=int)
    X = pd.concat([data[mv_col],industry_dummies], axis=1)
    X = sm.add_constant(X)
    y = data[factor_col]
    model = sm.OLS(y, X).fit()
    data['Traction_LUD']= model.resid
    # neutralized_factor = model.resid
    # return neutralized_factor, model
    return data

def run(start, end):
    warnings.filterwarnings('ignore')
    tar_date = get_tar_date(start,end)
    daily_df = feather.read_dataframe(DataPath.daily_path)[['DATE', 'TICKER', 'close']]
    daily_df=daily_df[daily_df.DATE>'20191229']
    daily_df.sort_values(['TICKER','DATE'],inplace=True)
    daily_df = process_na_stock(daily_df, 'close')
    daily_df['close_roll20']=daily_df.groupby('TICKER')['close'].shift(20)
    daily_df['ret_roll20']=daily_df['close']/daily_df['close_roll20']-1
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    daily_df = daily_df[daily_df.DATE.isin(tar_date)]

    # res=[]
    # for date in tqdm(tar_date):
    #     df=cal(date,daily_df[daily_df.DATE==date][['TICKER','ret_roll20']])
    #     res.append(df)
    res=Parallel(n_jobs=12)(delayed(cal)(date,daily_df[daily_df.DATE==date][['TICKER','ret_roll20']]) for date in tqdm(tar_date))
    res=pd.concat(res).reset_index(drop=True)
    # mkt_df=feather.read_dataframe(os.path.join(DataPath.to_df_path,'float_mv.feather')) #市值数据
    # citic_df=feather.read_dataframe(os.path.join(DataPath.to_df_path,'citic_code.feather')) #行业数据
    # res=res.merge(mkt_df,on=['DATE','TICKER'],how='inner').merge(citic_df,on=['DATE','TICKER'],how='inner')
    res.dropna(inplace=True)
    # 做截面反转、市值、行业中性化
    # res= res.groupby('DATE').apply(lambda x: neutralize_factor(x, 'exp_LUD', 'float_mv', 'citic_code')).reset_index(drop=True)
    # 重命名列
    res = res[['DATE', 'TICKER', 'exp_LUD']]
    # res = res[['DATE', 'TICKER', 'exp_LUD','Traction_LUD']]
    return [res]

def update_muli(filename,today,run,num=-50):
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
        res=run(start='20200101',end='20250929')
        # res=run(start='20200101',end='20250822')
        # res=run(start='20250828',end='20250829')
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                print(tmp)
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))
                # feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))

def update(today='20250929'):
    update_muli('exp_LUD.feather',today,run)

if __name__=='__main__':
    # run('20250102','20250107')
    update()