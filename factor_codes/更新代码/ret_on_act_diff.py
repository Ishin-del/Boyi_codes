

# diff=主动买-主动卖的量,选取diff最高/最低的10%的区间，表示投资者对该只股票异常热情/冷淡
# (也可以diff主动买或主动卖，看情绪变化激增或骤减的情况，这些区间挑选出来)
# order buy和实际成交之间的关系
# 以分钟回报率均值为因子
import os

from joblib import Parallel, delayed
from tqdm import tqdm

from tool_tyx.path_data import DataPath
import feather
import numpy as np
import pandas as pd
from tool_tyx.tyx_funcs import read_min_data, get_tar_date, process_na_stock, update_muli


def cal(date):
    """
    因子逻辑：
    diff=主动买-主动卖的量,选取diff最高/最低的10%的区间，表示投资者对该只股票异常热情/冷淡
    计算期间的ret均值
    :param date:
    :return:
    """
    df=read_min_data(date)[['TICKER','min','active_buy_volume','active_sell_volume','open','close']]
    df=df[(df['min']!=930)&(df['min']<1457)]
    df['min_ret']=df['close']/df['open']-1
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.sort_values(['TICKER','min'],inplace=True)
    df['buy_diff_sell']=df['active_buy_volume']-df['active_sell_volume']
    df['act_diff_self']=df.groupby('TICKER')['active_buy_volume'].diff()
    tmp1=df.groupby('TICKER')['min_ret'].mean()
    # ----------------------
    # 根据buy_diff_sell排序
    # 最高排名10%
    df.sort_values(['TICKER','buy_diff_sell'],ascending=[True,False],inplace=True)
    res1=help(df,24,tmp1,'retOnNetAct_top_24')
    # 最低排名10%
    df.sort_values(['TICKER','buy_diff_sell'],inplace=True)
    res2=help(df,24,tmp1,'retOnNetAct_bottom_24')
    # ----------------------
    # 根据act_diff_self排序
    df.sort_values(['TICKER','act_diff_self'],ascending=[True,False],inplace=True)
    res3=help(df,24,tmp1,'retOnBuyDiff_top_24')
    df.sort_values(['TICKER', 'act_diff_self'], inplace=True)
    res4 = help(df, 24, tmp1, 'retOnBuyDiff_bottom_24')
    tmp=res1.merge(res2,on=['TICKER'],how='inner').merge(res3,on=['TICKER'],
                                            how='inner').merge(res4,on=['TICKER'],how='inner')
    # tmp=res1.merge(res3,on=['TICKER'],how='inner')
    tmp['DATE']=date
    tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
    return tmp#[['DATE','TICKER','retOnVol_top_24']]

def help(df, top_num, tmp1, new_name):
    tmp = df.groupby('TICKER').head(top_num)  # 24,47
    tmp2 = tmp.groupby('TICKER')['min_ret'].mean()
    tmp2.replace([np.inf, -np.inf], np.nan, inplace=True)
    res = tmp2 - tmp1
    res = res.reset_index().rename(columns={'min_ret': new_name})  # 'vol_top_ratio'
    return res

def run(start,end):
    date_list=get_tar_date(start,end)
    # res=[]
    # for date in tqdm(date_list):
    #     tmp=cal(date)
    #     res.append(tmp)
    res=Parallel(12)(delayed(cal)(date) for date in tqdm(date_list))
    res=pd.concat(res).reset_index(drop=True)
    res.sort_values(['TICKER', 'DATE'], inplace=True)
    res.replace([np.inf, -np.inf], np.nan, inplace=True)
    res = process_na_stock(res, 'retOnNetAct_top_24')
    res['retOnNetAct_top_24_roll20'] = res.groupby('TICKER')['retOnNetAct_top_24'].rolling(20, 5).mean().values
    res['retOnNetAct_bottom_24_roll20'] = res.groupby('TICKER')['retOnNetAct_bottom_24'].rolling(20, 5).mean().values
    res['retOnBuyDiff_top_24_roll20'] = res.groupby('TICKER')['retOnBuyDiff_top_24'].rolling(20, 5).mean().values
    res['retOnBuyDiff_bottom_24_roll20'] = res.groupby('TICKER')['retOnBuyDiff_bottom_24'].rolling(20, 5).mean().values
    res.replace([np.inf, -np.inf], np.nan, inplace=True)
    res=res[['DATE','TICKER','retOnNetAct_top_24_roll20','retOnNetAct_bottom_24_roll20','retOnBuyDiff_top_24_roll20',
             'retOnBuyDiff_bottom_24_roll20']]
    # res=res[['DATE','TICKER','retOnNetAct_top_24_roll20','retOnBuyDiff_top_24_roll20']]
    return [res]
def update(today):
    update_muli('retOnNetAct_top_24_roll20.feather',today,run,-90)


def update_muli(filename,today,run,num=-50):
    if os.path.exists(os.path.join(DataPath.save_path_update,filename)):
    # if False:
        print('因子更新中')
        old=feather.read_dataframe(os.path.join(DataPath.save_path_update,filename))
        new_start=sorted(old.DATE.drop_duplicates().to_list())[num]
        res=run(start=new_start,end=today)
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                # feather.write_dataframe(tmp, os.path.join(r'C:\Users\admin\Desktop', col + '.feather'))
                old=feather.read_dataframe(os.path.join(DataPath.save_path_update,col+'.feather'))
                test=old.merge(tmp,on=['DATE','TICKER'],how='inner').dropna()
                test.sort_values(['TICKER','DATE'],inplace=True)
                tar_list = sorted(list(test.DATE.unique()))[-5:]
                test = test[test.DATE.isin(tar_list)]
                if np.isclose(test.iloc[:,2],test.iloc[:,3]).all():
                    tmp=tmp[tmp.DATE>old.DATE.max()]
                    old=pd.concat([old,tmp]).reset_index(drop=True).drop_duplicates()
                    print(old)
                    feather.write_dataframe(old,os.path.join(DataPath.save_path_update,col+'.feather'))
                    feather.write_dataframe(old,os.path.join(DataPath.factor_out_path,col+'.feather'))
                else:
                    print(test[~np.isclose(test.iloc[:,2],test.iloc[:,3])])
                    # tt=test[~np.isclose(test.iloc[:, 2], test.iloc[:, 3])]
                    # feather.write_dataframe(tt,r'C:\Users\admin\Desktop\tt.feather')
                    print('检查更新出错!')
                    exit()
    else:
        print('因子生成中')
        # res=run(start='20211201',end='20250801')
        res=run(start='20200101',end='20221231')
        # res=run(start='20250828',end='20250829')
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                print(tmp)
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))
                # feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))

if __name__=='__main__':
    # run('20250102','20250106')
    update('20251015')