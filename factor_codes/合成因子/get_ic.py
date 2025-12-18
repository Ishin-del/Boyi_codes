import feather
import os
import numpy as np
import pandas as pd
from tqdm import tqdm


dd=['20240206', '20240207', '20240208','20240926', '20240927', '20240930', '20241008']
# code_pool1=pd.read_csv(r'C:\Users\admin\Desktop\tickerpool1.csv',index_col=0)
code_pool1=pd.read_csv(r'C:\Users\admin\Desktop\tickerpool2.csv',index_col=0)
code_pool1['DATE']=code_pool1['DATE'].astype(str)
tar_list=os.listdir(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子')
res=[]
for file in tqdm(tar_list):
    tmp=feather.read_dataframe(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file}')
    the_col=list(np.setdiff1d(tmp.columns,['TICKER','DATE']))[0]
    tmp.sort_values(['TICKER','DATE'],inplace=True)
    tmp[the_col]=tmp.groupby('TICKER')[the_col].shift(1)
    tmp=tmp.merge(code_pool1,on=['TICKER','DATE'],how='inner')
    tmp=tmp[~tmp['DATE'].isin(dd)]
    ic=float(tmp[[the_col,'ret']].corr().iloc[0,1])
    ll=[file.replace('.feather',''),ic]
    print(ll)
    res.append(ll)

res=pd.DataFrame(res)
res.columns=['factor','ic_2']
res.to_csv(r'C:\Users\admin\Desktop\factor_ic_2.csv',index=False,encoding='gbk')
