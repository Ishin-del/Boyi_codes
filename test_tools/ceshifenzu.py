import gc
import time
import pandas as pd
import numpy as np
import feather
import os

from tqdm import tqdm
import warnings
import shutil
import subprocess
from typing import List
from datetime import timedelta, datetime
from collections import defaultdict



# datapath = r"F:\Quantzkh\Factor_calc\曹楚涵\SZ_model\SZ_VAE_model\VAE_prior\version_20250101\prediction.csv"
# datapath = r"D:\因子保存\structured_reversal_factor_new\factor"
# datapath = r"D:\因子保存\Risk_uncer"
datapath = r"C:\Users\admin\Desktop\test"
ret = feather.read_dataframe(r"\\192.168.1.210\tianyixin\calc6临时用\开盘买入出场四收益.feather")

ret = ret[['TICKER','DATE','平均卖出收益_1100_1400']]
ret['ret'] = ret['平均卖出收益_1100_1400']

ret['ret'] = ret['ret'].clip(-0.4, 0.4)  # 限制在[-0.4, 0.4]
ret['ret'] -= 0.0007  # 统一减去一个基准值
# ret = ret[['TICKER', 'DATE', '平均卖出收益_1100_1400']]
# ret = ret.rename(columns={'平均卖出收益_1100_1400':'ret'})
ret = ret[['TICKER', 'DATE', 'ret']]
# ret.loc[ret['ret'] > 0.4,'ret'] = 0.4

ret['DATE'] = ret['DATE'].astype(str)
# ret = ret[ret['DATE'] >= final['DATE'].min()]
# path = r"F:\vp_zkh\lgb_small\hgbt\ckpt\20250102\hgbt_trial_18\add_test_predict.feather" # 10  0.008856

def get_qcut(x,usename):
    x['bins'] = pd.qcut(x[usename], 10, labels=False,duplicates='drop') + 1
    return x

for i in os.listdir(datapath):
    warnings.filterwarnings(action='ignore')
    if i.endswith('.feather'):
        final = feather.read_dataframe(os.path.join(datapath,i))
        # final = final.reset_index(drop=False)
            # final = pd.read_csv(datapath)
        final['DATE'] = final['DATE'].astype(str)
        zz = pd.merge(ret,final,how='inner',on=['TICKER','DATE'])
        usename = list(np.setdiff1d(final.columns,['TICKER','DATE']))[0]
        # bins = pd.qcut(zz[usename], 10)

        zz["rank"] = zz.groupby("DATE")[usename].rank(pct=True)  # 0-1的百分比排名
        zz["rank"] = zz[usename].rank(pct=True)  # 0-1的百分比排名
        zz["group"] = zz["rank"].apply(lambda x:np.floor((x - 0.0000001) * 10) + 1)

        # zz = zz.groupby('DATE').apply(get_qcut,usename).reset_index(drop=True)
        res = zz.groupby(['DATE','group'])['ret'].mean().reset_index(drop=False)
        res = res.groupby('group')['ret'].mean().reset_index(drop=False)
        cnt = zz.groupby('group').count()[usename].reset_index(drop=False)
        res  =pd.merge(res,cnt,how='left',on=['group']).reset_index(drop=True)
        print(i)
        print(res)
