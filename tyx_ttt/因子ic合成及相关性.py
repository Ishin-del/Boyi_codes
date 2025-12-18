import os

import feather
import pandas as pd
from tqdm import tqdm


def get_res(factor_path,save_path,ic_path):
    tar_lst1=os.listdir(factor_path)
    res=[]
    for file in tqdm(tar_lst1):
        # 拉取因子
        tmp=feather.read_dataframe(os.path.join(factor_path,file)).set_index(['TICKER','DATE'])
        res.append(tmp)
    res=pd.concat(res,axis=1)
    res=res.corr().reset_index()
    res.to_csv(os.path.join(save_path,'factor_corr.csv'),index=False,encoding='gbk')
    # -------------------------
    # tar_lst2=os.listdir(ic_path)
    # res=[]
    # for file in tqdm(tar_lst2):
    #     # 拉取因子ic
    #     tmp=pd.read_csv(os.path.join(ic_path,file),encoding='gbk')
    #     tmp['factor']=file.replace('.csv','')
    #     res.append(tmp[tmp['bins']=='ic'])
    # res=pd.concat(res).reset_index(drop=True)
    # res.to_csv(os.path.join(save_path,'factor_ic.csv'),index=False,encoding='gbk')

if __name__=='__main__':
    get_res(factor_path=r'\\192.168.1.210\Factor_Storage\田逸心\原始数据\拟使用量价因子',
            save_path=r'D:\tyx\因子情况',
            ic_path=r'\\192.168.1.210\因子测试结果\田逸心\原因子增量回测22-24')
