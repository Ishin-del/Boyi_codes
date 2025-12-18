import os
import random
import warnings
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm

def tar_codes(file):
    start,end='20220101','20241231'
    warnings.filterwarnings('ignore')
    tmp=feather.read_dataframe(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file}') #拉因子
    tmp=tmp[(tmp['DATE']>=start)&(tmp['DATE']<=end)].reset_index(drop=True)
    bt=pd.read_csv(os.path.join(bt_path,file.replace('feather','csv'))) #拉回测

    the_col=np.setdiff1d(tmp.columns,['DATE','TICKER'])
    tmp.dropna(inplace=True)
    tmp.sort_values(['DATE',the_col.tolist()[0]],inplace=True)
    tmp.reset_index(drop=True,inplace=True)
    if bt.tail(1).iloc[0,1]<0 and (bt.head(10)['分层收益']<0).any(): #负向，取小值
        tmp=tmp.groupby('DATE',as_index=False).apply(lambda x: x.head(int(len(x)*0.9))).reset_index(drop=True)
    elif bt.tail(1).iloc[0,1]>0 and (bt.head(10)['分层收益']<0).any():
        tmp = tmp.groupby('DATE',as_index=False).apply(lambda x: x.tail(int(len(x) * 0.9))).reset_index(drop=True)
    tmp['DATE'] = tmp['DATE'].astype(str)
    return tmp

if __name__=='__main__':
    file_list=os.listdir(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子')

    bt_path=r'\\Desktop-79nue61\因子测试结果\田逸心\原因子增量回测22-24'
    # start='20250101'
    # end='20250901'
    for file in tqdm(file_list):
        if file == 'VaR_5%_v2.feather':
            continue
        if os.path.exists(fr'C:\Users\admin\Desktop\res_file_25\{file.replace('.feather','.csv')}'):
            continue
        df=tar_codes(file)
        ret_df=feather.read_dataframe(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\因子组合收益return.feather')
        df=df.merge(ret_df,on=['DATE','TICKER'],how='left')
        df.sort_values(['TICKER','DATE'],inplace=True)
        df['return']=df.groupby('TICKER')['return'].shift(-1)
        # res=df.groupby('DATE').agg({'TICKER':'size','return':'mean'}).reset_index()
        dd=['20240206', '20240207', '20240208','20240926', '20240927', '20240930', '20241008']
        df=df[~df['DATE'].isin(dd)]
        # print(res)
        # feather.write_dataframe(res,r'C:\Users\admin\Desktop\res.feather')
        df.to_csv(fr'C:\Users\admin\Desktop\res_file_25\{file.replace('.feather','.csv')}')


# ll=os.listdir(r'C:\Users\admin\Desktop\res_file_25')
# for f in ll:
#     tmp=pd.read_csv(fr'C:\Users\admin\Desktop\res_file\{f}')
#     m=tmp['return'].mean()
#     if m>0.0007:
#         print(f,m)