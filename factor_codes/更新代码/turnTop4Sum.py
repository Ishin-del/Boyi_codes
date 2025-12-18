
"""
1. 对个股过去20日的收盘价做排序，选出收盘价最低的4天，计算换手率的和与20日换手率总和比值，比值越大说明低位放量现象越显著；
2. 对个股过去20日的收盘价做排序，选出收盘价最低的4天，计算波动率均值与20日波动率均值比值，比值越大说明低位高波动现象越显著；
3. 对个股过去20日的收盘价做排序，选出收盘价最高的4天，计算换手率的和与20日换手率总和比值，比值越大说明高位放量现象越显著；
4. 对个股过去20日的收盘价做排序，选出收盘价最高的4天，计算波动率均值与20日波动率均值比值，比值越大说明高位高波动现象越显著；
"""
import time
import feather
import os
from tool_tyx.path_data import DataPath
import numpy as np
import pandas as pd
from tool_tyx.tyx_funcs import process_na_stock


def help(x,ascending=True):
    n = len(x)
    windows = [x.iloc[i - 19:i + 1] if i >= 19 else None for i in range(n)]
    results = []
    for window in windows:
        if window is None or len(window) != 20:
            results.append([np.nan, np.nan])
        else:
            sorted_window = window.sort_values('close', ascending=ascending)
            turn_sum = sorted_window['turnover'].head(4).sum()
            vol_std = sorted_window['daily_ret'].head(4).std()
            results.append([turn_sum, vol_std])
    return pd.DataFrame(results)

def run(start='20250102',end='20250401'):
    print(start,end)
    daily_df = feather.read_dataframe(DataPath.daily_path)[['DATE','TICKER','close','amount']]
    daily_df.sort_values(['TICKER', 'DATE'], inplace=True)
    daily_df['daily_ret'] = daily_df.groupby('TICKER')['close'].pct_change()
    daily_df=daily_df[(daily_df['DATE']>=start)&(daily_df['DATE']<=end)]
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    float_mv = feather.read_dataframe(os.path.join(DataPath.to_df_path, 'float_mv.feather'))
    daily_df = daily_df.merge(float_mv, on=['DATE', 'TICKER'], how='inner')
    daily_df['turnover'] = daily_df['amount'] / daily_df['float_mv']
    daily_df.drop(columns=['amount','float_mv'],inplace=True)
    daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
    daily_df=process_na_stock(daily_df,'turnover')
    # ----------------
    daily_df['turnover_20_sum'] = daily_df.groupby('TICKER')['turnover'].transform(lambda x: x.rolling(20, 5).sum())
    daily_df['volatility_20'] = daily_df.groupby('TICKER')['daily_ret'].transform(lambda x: x.rolling(20, 5).std())
    daily_df.set_index(['TICKER','DATE'],inplace=True)
    # t1=time.time()
    tmp1=daily_df.groupby('TICKER').apply(help).reset_index(drop=True).rename(columns={0:'turnTail4Sum',1:'retTail4Std'})
    tmp1.index=daily_df.index
    tmp2=daily_df.groupby('TICKER').apply(help,ascending=False).reset_index(drop=True).rename(columns={0:'turnHead4Sum',1:'retHead4Std'})
    tmp2.index=daily_df.index
    # t2=time.time()
    # print((t2-t1)/60)
    daily_df=daily_df.merge(tmp1.reset_index(),on=['DATE','TICKER'],how='inner').merge(tmp2.reset_index(),on=['DATE','TICKER'],how='inner')
    daily_df['turnTail4Sum_ratio']=daily_df['turnTail4Sum']/daily_df['turnover_20_sum']
    daily_df['turnHead4Sum_ratio']=daily_df['turnHead4Sum']/daily_df['turnover_20_sum']
    # daily_df['retTail4Std_ratio']=daily_df['retTail4Std']/daily_df['volatility_20']
    # daily_df['retHead4Std_ratio']=daily_df['retHead4Std']/daily_df['volatility_20']
    daily_df=daily_df[['DATE', 'TICKER','turnTail4Sum_ratio','turnHead4Sum_ratio']] #,'retTail4Std_ratio','retHead4Std_ratio'
    # print(daily_df)
    return [daily_df]

def update(today='20251016'):
    update_muli('turnTail4Sum_ratio.feather',today,run)

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
        # res=run(start='20200101',end='20221231')
        res=run(start='20220101',end='20250101')
        # res=run(start='20250828',end='20250829')
        for df in res:
            for col in df.columns[2:]:
                tmp=df[['DATE','TICKER',col]]
                print(tmp)
                # feather.write_dataframe(tmp,os.path.join(DataPath.save_path_old,col+'.feather'))
                # feather.write_dataframe(tmp,os.path.join(DataPath.save_path_update,col+'.feather'))
                feather.write_dataframe(tmp,os.path.join(r'C:\Users\admin\Desktop\test',col+'.feather'))
if __name__=='__main__':
    t1=time.time()
    update()
    # run()
    t2=time.time()
    print((t2-t1)/60)
