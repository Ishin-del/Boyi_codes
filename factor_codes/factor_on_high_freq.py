import os
import feather
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from tqdm import tqdm
from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import get_tar_date, update_muli


# 中信建投：高频量价选股因子初探：  ——因子深度研究系列
def get_weight():
    res=[]
    for i in range(1, 6):
        tmp=1 - (i - 1) / 5
        res.append(tmp)
    return res

def read_snap_data(code,path,date):
    tmp=feather.read_dataframe(os.path.join(path,code))
    tmp=tmp[(tmp['DateTime']>=int(date+'093100'))&(tmp['DateTime']<int(date+'145800'))]
    tmp = tmp[['SecurityID', 'DateTime', 'BidPrice0', 'BidPrice1', 'BidPrice2', 'BidPrice3', 'BidPrice4', 'BidOrderQty0',
             'BidOrderQty1', 'BidOrderQty2', 'BidOrderQty3', 'BidOrderQty4', 'OfferPrice0', 'OfferPrice1','OfferPrice2',
             'OfferPrice3', 'OfferPrice4', 'OfferOrderQty0', 'OfferOrderQty1', 'OfferOrderQty2', 'OfferOrderQty3',
             'OfferOrderQty4']]
    return tmp

def cal(date):
    if os.path.exists(fr'D:\因子输出\tmp\{date}.feather'):
        tmp=feather.read_dataframe(fr'D:\因子输出\tmp\{date}.feather')
        return tmp
    if date[:4]=='2022':
        path=os.path.join(DataPath.feather_2022,date,'Snapshot')
        code_files=os.listdir(path)
    elif date[:4]=='2023':
        path=os.path.join(DataPath.feather_2023,date,'Snapshot')
        code_files = os.listdir(path)
    elif date[:4] == '2024':
        path=os.path.join(DataPath.feather_2024,date,'Snapshot')
        code_files = os.listdir(path)
    elif date[:4] == '2025':
        path=os.path.join(DataPath.feather_2025,date,'Snapshot')
        code_files = os.listdir(path)
    # df=[]
    # for code in code_files:
    #     tmp=read_snap_data(code,path,date)
    #     df.append(tmp)
    df=Parallel(n_jobs=25)(delayed(read_snap_data)(code,path,date) for code in tqdm(code_files))
    df=pd.concat(df).reset_index(drop=True)

    df.sort_values(['SecurityID','DateTime'],inplace=True)
    # 因子一----------------------------
    df[f'VwB'] = (1 * df['BidOrderQty0'] + 0.8 * df['BidOrderQty1'] + 0.6 * df['BidOrderQty2'] + 0.4 * df['BidOrderQty3'] + 0.199 * df['BidOrderQty4']) / 3
    df[f'VwA'] = (1 * df['OfferOrderQty0'] + 0.8 * df['OfferOrderQty1'] + 0.6 * df['OfferOrderQty2'] + 0.4 * df['OfferOrderQty3'] + 0.199 * df['OfferOrderQty4']) / 3

    for col in df.columns[2:]:
        df[col+'_help']=df.groupby('SecurityID')[col].shift(1)

    df[f'VB'] = np.where(df[f'BidPrice{0}'] < df[f'BidPrice{0}_help'], 0,np.where(df[f'BidPrice{0}'] > df[f'BidPrice{0}_help'], df[f'VwB'], df[f'VwB_help']-df[f'VwB']))
    df[f'VA']=np.where(df[f'OfferPrice{0}'] < df[f'OfferPrice{0}_help'], 0,np.where(df[f'OfferPrice{0}'] > df[f'OfferPrice{0}_help'], df[f'VwA'], df[f'VwA_help']-df[f'VwA']))
    df['VOI2']=df['VB']-df['VA']
    # 因子二---------------------------
    df['OIR']=(df['VB']-df['VA'])/(df['VB']+df['VA'])
    # todo:因子三---------------------------

    # ====================================
    # todo:inf
    # 因子高频转低频：
    df['VOI2_adjust']=df.groupby('DateTime')['VOI2'].transform(lambda x: (x-x.mean())/x.std())
    df['OIR_adjust']=df.groupby('DateTime')['OIR'].transform(lambda x: (x-x.mean())/x.std())
    tmp1=df.groupby('SecurityID')['VOI2_adjust'].mean().reset_index().rename(columns={'SecurityID':'TICKER'})
    tmp2=df.groupby('SecurityID')['OIR_adjust'].mean().reset_index().rename(columns={'SecurityID':'TICKER'})
    tmp=tmp1.merge(tmp2,on='TICKER',how='inner')
    tmp['DATE']=date
    tmp=tmp[['TICKER','DATE','VOI2_adjust','OIR_adjust']]
    feather.write_dataframe(tmp,fr'D:\因子输出\tmp\{date}.feather')
    return tmp

def run(start,end):
    tar_date=get_tar_date(start,end)
    df=[]
    for date in tar_date:
        print(f'{date}数据计算中...')
        tmp=cal(date)
        df.append(tmp)
    # df=Parallel(n_jobs=20)(delayed(cal)(date) for date in tqdm(tar_date))
    df=pd.concat(df).reset_index(drop=True)
    return [df]

def update(today='20251028'):
    update_muli('VOI2_adjust',today,run)


if __name__=='__main__':
    # cal('20230105')
    # run('20230103','20230105')
    # run('20220101','20250729')
    update('20250729')
