import os
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm

ll=os.listdir(r'\\192.168.1.210\Factor_Storage\田逸心\原始数据\拟使用量价因子')
res=[]
for f in tqdm(ll):
    tmp=feather.read_dataframe(fr'\\192.168.1.210\Factor_Storage\田逸心\原始数据\拟使用量价因子\{f}')
    tar_col = list(np.setdiff1d(tmp.columns,['DATE','TICKER']))
    ret=feather.read_dataframe(r'C:\Users\admin\Desktop\ret.feather')
    tmp=tmp.merge(ret,on=['DATE','TICKER'],how='inner')
    tmp.sort_values(['TICKER','DATE'],inplace=True)
    tmp[tar_col]=tmp.groupby('TICKER')[tar_col].shift(1)
    res.append([f.replace('.feather',''),tmp[[tar_col[0],'平均卖出收益_1100_1400']].corr().iloc[0,1]])

res=pd.DataFrame(res)
res.columns=['factor','ic']
res.to_csv(r'C:\Users\admin\Desktop\res.csv',index=False,encoding='gbk')