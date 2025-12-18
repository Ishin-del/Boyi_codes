import os
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
from tool_tyx.path_data import DataPath

def select_factors(path=DataPath.factor_path):
    # files=os.listdir(path)
    # df=pd.DataFrame()
    # for f in tqdm(files):
    #     tmp=feather.read_dataframe(os.path.join(path,f))
    #     tmp=tmp[tmp.DATE>='20220101']
    #     if df.empty:
    #         df=tmp
    #     else:
    #         df=df.merge(tmp,on=['TICKER','DATE'],how='inner')
    # print(df)
    # feather.write_dataframe(df,r'C:\Users\admin\Desktop\factors.feather')
    df=feather.read_dataframe(r'C:\Users\admin\Desktop\factors.feather')
    # print(df.iloc[:, 2:].corr())
    # corr_matrix = df.iloc[:, 2:].corr()
    # corr_matrix.to_csv(r'C:\Users\admin\Desktop\corr.csv')
    # 选'AmtPerTrd_rolling20','elasticity_adjust_std20','GTR','Mom_bigOrder_70_rolling20','open_buy_will_ratio_roll20','open_buy_will_str_roll20','OvernightSmart20','RSkew_daily_y','DTGD','随波逐流'
    # df=df[['DATE','TICKER','AmtPerTrd_rolling20','elasticity_adjust_std20','GTR','Mom_bigOrder_70_rolling20','open_buy_will_ratio_roll20','open_buy_will_str_roll20',
    #        'OvernightSmart20','RSkew_daily_y','DTGD','随波逐流']]
    df=df[['DATE','TICKER','tree_soldier_ret','tree_soldier_vol','勇攀高峰_pure','收益公平因子_roll20','flow_elasticity']]
    return df

def orth(data):
    ## step5:构建对称正交变换函数
    data_ = data.copy()  # 创建副本不影响原数据
    col = data_.columns.tolist()
    # F = np.mat(data_[col])  # 将数据框转化为矩阵
    F = np.asmatrix(data_[col])  # 将数据框转化为矩阵
    M = F.T @ F  # 等价于 (F.shape[0] - 1) * np.cov(F.T)
    a, U = np.linalg.eig(M)  # a为特征值，U为特征向量
    D_inv = np.linalg.inv(np.diag(a))
    S = U @ np.sqrt(D_inv) @ U.T
    data_[col] = data_[col].dot(S)
    return data_

def process_data():
    df=select_factors()
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    #方向：
    # df['AmtPerTrd_rolling20']*=-1
    # df['elasticity_adjust_std20']*=-1
    # df['GTR']*=-1
    # df['Mom_bigOrder_70_rolling20']*=-1
    # df['RSkew_daily_y']*=-1
    # df['open_buy_will_ratio_roll20']*=-1
    # df['open_buy_will_str_roll20']*=-1
    # df['OvernightSmart20']*=-1
    # df['DTGD']*=-1
    # df['随波逐流']*=-1
    df['tree_soldier_ret']*=-1
    df['tree_soldier_vol']*=-1
    # 标准化
    df[df.columns[2:]]=df.iloc[:,2:].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    df.replace(np.nan,0,inplace=True)
    # 正交
    df[df.columns[2:]]=orth(df.iloc[:,2:])
    df['comp_factor']=df.iloc[:, 2:].sum(axis=1)
    # print(df)
    df=df[['DATE','TICKER','comp_factor']]
    # feather.write_dataframe(df,r'C:\Users\admin\Desktop\test\test.feather')
    df.sort_values(['DATE','comp_factor'],ascending=[True,True],inplace=True)
    print(df)
    tmp=df.groupby('DATE')[['DATE','TICKER']].head(50).reset_index(drop=True)
    tmp['weight']=0.01
    print('---------------------')
    print(tmp)
    feather.write_dataframe(df, r'C:\Users\admin\Desktop\res.feather')


if __name__=='__main__':
    process_data()
