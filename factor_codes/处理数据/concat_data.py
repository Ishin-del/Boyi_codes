import os
import feather
import pandas as pd
from joblib import delayed,Parallel
from tqdm import tqdm


def cal_data(code,date,label):
    df = feather.read_dataframe(fr'\\192.168.1.7\{label}\data_feather\{date}\Snapshot\{code}')
    tar_help=[col for col in df.columns if col.startswith('Bid') or col.startswith('Offer')][:-2]
    # tar_help=[col for col in tar_help if col[-1].isalnum()]
    tar_cols = ['SecurityID', 'DateTime'] + tar_help
    df = df[tar_cols]
    df = df[df['DateTime'] < int(date + '092500')]
    return df

def get_data(date,label):
    path=fr'\\192.168.1.7\{label}\data_feather\{date}\Snapshot'
    # for code in os.listdir(path):
    #     code_path=fr'\\192.168.1.7\data04\data_feather\{date}\Snapshot\{code}'
    df=Parallel(n_jobs=12)(delayed(cal_data)(code,date,label) for code in tqdm(os.listdir(path),desc=f'{date}'))
    df=pd.concat(df).reset_index(drop=True)
    feather.write_dataframe(df,fr'D:\tyx\早盘集合竞价截取数据\{date}.feather')
    print(df)

def read_files(pp,label):
    ll=os.listdir(pp)
    for date in ll:
        if os.path.exists(fr'D:\tyx\早盘集合竞价截取数据\{date}.feather'):
            print(f'{date}数据已存在')
            continue
        get_data(date,label)

if __name__=='__main__':
    pp=r'\\192.168.1.7\data01\data_feather'
    pp1=r'\\192.168.1.7\data02\data_feather'
    pp2=r'\\192.168.1.7\data03\data_feather'
    pp3=r'\\192.168.1.7\data04\data_feather'
    read_files(pp, 'data01')
    read_files(pp1, 'data02')
    read_files(pp2, 'data03')
    read_files(pp3, 'data04')