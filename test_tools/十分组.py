import os
import warnings

import feather
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm


def cal_factors_validation(x, factor_name):
    d = x['DATE'].values[0]
    if 'DATE' in x.columns:
        x = x.drop(['DATE'], axis=1)
    if 'TICKER' in x.columns:
        x = x.drop(['TICKER'], axis=1)
    res = x.corr(method='spearman')
    res = res[[factor_name]]
    res = res.T
    res = res.drop(factor_name, axis=1)
    res = res.reset_index(drop=True)
    res['DATE'] = d
    return res

def group_mean(factor_name='',bin_num=10):
    warnings.filterwarnings('ignore')
    aa=feather.read_dataframe(rf'\\192.168.1.210\Factor_Storage\田逸心\原始数据\拟使用量价因子\{factor_name}')
    aa=aa[aa['TICKER'].str.endswith('.SZ')]
    aa.dropna(inplace=True)
    factor_name=factor_name.replace('.feather','')
    ret=feather.read_dataframe(r'C:\Users\admin\Desktop\ret.feather')
    # ret.loc[ret['平均卖出收益_1100_1400'] > 0.4, '平均卖出收益_1100_1400'] = 0.4
    ret['平均卖出收益_1100_1400'] = ret['平均卖出收益_1100_1400'].clip(-0.4, 0.4)  # 限制在[-0.4, 0.4]
    ret['平均卖出收益_1100_1400'] -= 0.0007  # 统一减去一个基准值
    ret['DATE'] = ret['DATE'].astype(str)
    ret.rename(columns={'平均卖出收益_1100_1400':'Return'},inplace=True)
    factor_data=ret.merge(aa,on=['DATE','TICKER'],how='inner')
    usename = list(np.setdiff1d(aa.columns, ['TICKER', 'DATE']))[0]
    factor_data.sort_values(['TICKER','DATE'],inplace=True)
    factor_data[usename]=factor_data.groupby('TICKER')[usename].shift(1)
    # todo:-----------------------------------
    factor_data = factor_data.dropna(subset=['Return']).reset_index(drop=True)
    ic_results = factor_data.groupby('DATE').apply(cal_factors_validation,factor_name=factor_name).reset_index(drop=True)
    bins = {}
    for key, series in factor_data.set_index('TICKER').groupby('DATE')[factor_name]:
        try:
            bins[key] = pd.qcut(series, q=bin_num, labels=range(1, bin_num + 1))
        except:
            print(factor_name+'数据出错！')
            return
    # df_bins = pd.melt(pd.DataFrame(bins), ignore_index=False, value_name='bins',
    #                   var_name='DATE').reset_index(drop=False)
    df_bins=pd.concat(bins,names=['DATE','index']).reset_index()
    df_bins.columns=['DATE','TICKER','bins']
    factor_data.reset_index(drop=True, inplace=True)
    factor_data = pd.merge(factor_data, df_bins, on=['TICKER', 'DATE'], how='left')
    factor_data['exchange_date'] = factor_data['DATE']
    factor_data_split = factor_data.groupby('bins').agg({'Return': ['mean', 'std']}).reset_index(drop=False)
    factor_data_split.columns = ['bins', 'Return_mean', 'Return_std']
    factor_data_split['IR_split_by_bins'] = factor_data_split['Return_mean'] / factor_data_split['Return_std']
    split_by_date = factor_data.groupby(['DATE', 'bins']).agg({'Return': 'mean'}).reset_index(drop=False)
    split_by_date['Return'] = split_by_date['Return'] / 100 + 1
    split_by_date['Return_cumprod'] = split_by_date.groupby('bins')['Return'].cumprod().reset_index(drop=True)
    mean_ic = np.mean(ic_results['Return'])
    df_factor = factor_data.dropna(subset=[factor_name])
    pct_all_dict = {}

    for i in range(1, bin_num + 1):
        pct_all_dict[i] = {}

    for trade_date, df_date in df_factor.groupby('DATE'): #, desc='计算各层的平均收益')
        for bin in range(1, bin_num + 1):
            df_temp = df_date[df_date['bins'] == bin]
            # df_temp.drop_duplicates(subset = ['TICKER'], inplace = True)
            return_data = df_temp['Return']
            # print(df_temp,return_data)
            if return_data.isna().min():
                pass
            else:
                pct_all_dict[bin][trade_date] = return_data.mean() + 1
    df_exchange_return = pd.DataFrame(pct_all_dict)
    tmp = df_exchange_return.mean() - 1
    tmp = tmp.reset_index().rename(columns={'index': 'bins', 0: '分层收益'})
    ic_row = pd.DataFrame([['ic', mean_ic]], columns=tmp.columns)
    tmp = pd.concat([tmp, ic_row], ignore_index=True)
    path=fr'\\192.168.1.210\因子测试结果\田逸心\竞价数据_sz'
    os.makedirs(path,exist_ok=True)
    print(tmp)
    tmp.to_csv(os.path.join(path,factor_name+'.csv'))
    return tmp

if __name__=='__main__':
    ll=os.listdir(r'\\192.168.1.210\Factor_Storage\田逸心\原始数据\拟使用量价因子')
    # Parallel(n_jobs=12)(delayed(group_mean)(f) for f in tqdm(ll))
    for f in ll:
        if not os.path.exists(fr'\\192.168.1.210\因子测试结果\田逸心\竞价数据_sz\{f.replace('.feather','')}.csv'):
            group_mean(f)
        else:
            print(f'{f}数据已生成')
    # todo:ctr,exp_LUD,ideal_v_diff,ideal_v_high,ideal_v_low,nlsize_risk_factor_adjust,OvernightSmart20,随波逐流数据
