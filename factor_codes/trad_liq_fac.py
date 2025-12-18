# @Author: Yixin Tian
# @File: trad_liq_fac.py
# @Date: 2025/10/28 14:21
# @Software: PyCharm

import feather
import os
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tool_tyx.tyx_funcs import get_tar_date, update_muli
from tool_tyx.path_data import DataPath

# 中信建投：买卖报单流动性因子构建： ——因子深度研究系列

def read_snap_data(code,path,date):
    tmp=feather.read_dataframe(os.path.join(path,code))
    tmp=tmp[(tmp['DateTime']>=int(date+'093100'))&(tmp['DateTime']<int(date+'145800'))]
    tmp = tmp[['SecurityID', 'DateTime','BidPrice0','BidPrice1','BidPrice2','BidPrice3','BidPrice4',
                                    'BidOrderQty0','BidOrderQty1', 'BidOrderQty2', 'BidOrderQty3', 'BidOrderQty4',
                                    'OfferPrice0', 'OfferPrice1', 'OfferPrice2','OfferPrice3', 'OfferPrice4',
                                    'OfferOrderQty0','OfferOrderQty1','OfferOrderQty2','OfferOrderQty3','OfferOrderQty4']]
    return tmp

def cal(date):
    try:
        tmp = feather.read_dataframe(fr'D:\因子输出\tmp2\{date}.feather')
        return tmp
    except:
        return
    if os.path.exists(fr'D:\因子输出\tmp2\{date}.feather'):
        tmp = feather.read_dataframe(fr'D:\因子输出\tmp\{date}.feather')
        return tmp
    if date[:4] == '2022':
        path = os.path.join(DataPath.feather_2022, date, 'Snapshot')
        code_files = os.listdir(path)
    elif date[:4] == '2023':
        path = os.path.join(DataPath.feather_2023, date, 'Snapshot')
        code_files = os.listdir(path)
    elif date[:4] == '2024':
        path = os.path.join(DataPath.feather_2024, date, 'Snapshot')
        code_files = os.listdir(path)
    elif date[:4] == '2025':
        path = os.path.join(DataPath.feather_2025, date, 'Snapshot')
        code_files = os.listdir(path)
    df = Parallel(n_jobs=25)(delayed(read_snap_data)(code, path, date) for code in tqdm(code_files,desc=f'{date}数据计算中'))
    df = pd.concat(df).reset_index(drop=True)

    df['M_0'] = (df[f'BidPrice0'] + df[f'OfferPrice0']) / 2  # 买卖单均值,限价交易成本
    for i in range(5):
        df[f'OffAmn_{i}']=df[f'OfferPrice{i}']*df[f'OfferOrderQty{i}']
        df[f'BidAmn_{i}']=df[f'BidPrice{i}']*df[f'BidOrderQty{i}']
    df['OfferQtyTotal']=df['OfferOrderQty0']+df['OfferOrderQty1']+df['OfferOrderQty2']+df['OfferOrderQty3']+df['OfferOrderQty4']
    df['BidQtyTotal']=df['BidOrderQty0']+df['BidOrderQty1']+df['BidOrderQty2']+df['BidOrderQty3']+df['BidOrderQty4']
    df['DolVol_A']=df['OffAmn_0']+df['OffAmn_1']+df['OffAmn_2']+df['OffAmn_3']+df['OffAmn_4']
    df['DolVol_B']=df['BidAmn_0']+df['BidAmn_1']+df['BidAmn_2']+df['BidAmn_3']+df['BidAmn_4']
    df['VWAP_A']=(df['DolVol_A'])/df['OfferQtyTotal'] #卖单市价交易成本
    df['VWAP_B']=(df['DolVol_B'])/df['BidQtyTotal'] #买单市价交易成本
    df['VWAPM_A']=df['VWAP_A']/df['M_0']-1
    df['VWAPM_B']=-df['VWAP_B']/df['M_0']+1
    df['MCI_A']=df['VWAPM_A']/df['DolVol_A'] #因子一：卖单流动性因子
    df['MCI_B']=df['VWAPM_B']/df['DolVol_B'] #因子一：买单流动性因子
    df['MCI_IMB']=(df['MCI_B']-df['MCI_A'])/(df['MCI_B']+df['MCI_A'])
    # ---------------------------------------------------------
    # 高频转低频
    # 标准化
    df['MCI_A_adj'] = df.groupby('DateTime')['MCI_A'].transform(lambda x: (x-x.mean())/x.std())
    df['MCI_B_adj'] = df.groupby('DateTime')['MCI_B'].transform(lambda x: (x-x.mean())/x.std())
    df['MCI_IMB_adj']=df.groupby('DateTime')['MCI_IMB'].transform(lambda x: (x-x.mean())/x.std())
    # 转日频
    tmp=df.groupby('SecurityID').agg({'MCI_A_adj':'mean','MCI_B_adj':'mean','MCI_IMB_adj':'mean'}).reset_index().rename(columns={'SecurityID':'TICKER'})
    tmp['DATE']=date
    # print(tmp)
    os.makedirs(r'D:\因子输出\tmp2',exist_ok=True)
    feather.write_dataframe(tmp, fr'D:\因子输出\tmp2\{date}.feather')
    return tmp

def run(start, end):
    tar_date = get_tar_date(start, end)
    df = []
    for date in tqdm(tar_date):
        tmp = cal(date)
        df.append(tmp)
    # df=Parallel(n_jobs=20)(delayed(cal)(date) for date in tqdm(tar_date))
    df = pd.concat(df).reset_index(drop=True)
    return [df]

def update(today):
    update_muli('MCI_A_adj.feather',today,run)

if __name__=='__main__':
    update('20250729')
    # run('20220101', '20250729')