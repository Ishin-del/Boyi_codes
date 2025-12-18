from tool_tyx.data_path import DataPath
import os
import feather
import numpy as np
import pandas as pd

def check_factor_daily(file_name,FactorClass:object):
    save_path_old = DataPath.save_path_old
    save_path_update = DataPath.save_path_update
    flag = True
    while flag:
        if os.path.exists(os.path.join(save_path_update, file_name)):
            old_df = feather.read_dataframe(os.path.join(save_path_update, file_name))
            old_date_list = old_df.DATE.drop_duplicates().to_list()
            old_date_list = sorted(old_date_list)
            new_start = old_date_list[old_date_list.index(old_df.DATE.max()) - 20]
            print(f'因子{file_name.replace('.feather','')}更新中：')
            data = FactorClass( end='20250801').run() # todo: 每次更新检查，改end
            print('------------------------')
            test_df = old_df.merge(data, on=['DATE', 'TICKER'], how='inner')
            print(test_df)
            if np.isclose(test_df.iloc[:, 2], test_df.iloc[:, 3]).all():
                new_df = data[data.DATE > old_df.DATE.max()]
                full_df = pd.concat([old_df, new_df]).reset_index(drop=True)
                feather.write_dataframe(full_df, os.path.join(save_path_update, file_name))
                # todo:
                # feather.write_dataframe(full_df,os.path.join(save_path_old,'vcv_daily.feather'))
            else:
                print('检查更新，数据有问题！')
                exit()
            flag = False
        else:
            print(f'新因子{file_name.replace('.feather','')}生成中：')
            data = FactorClass(end='20221231') .run() # 第一次生成因子
            feather.write_dataframe(data, os.path.join(save_path_old, file_name))
            feather.write_dataframe(data, os.path.join(save_path_update, file_name))

# --------------------------------
p=r'\\Desktop-79nue61\因子测试结果\田逸心\20250820_中证2000_20220104_20250701_1'
for f in os.listdir(p):
    print(f)
    tmp=pd.read_excel(os.path.join(p,f,f.replace('half','')+'.xlsx')).T
    print(tmp)
    print('------------------------')



t1=feather.read_dataframe(r'D:\tyx\检查更新\2021.6-2025.8\Amt_netInFlow_bigOrder_ratio_24_rolling20.feather')
t2=feather.read_dataframe(r'C:\Users\admin\Desktop\Amt_netInFlow_bigOrder_ratio_24_rolling20.feather')
t1.columns=['DATE','TICKER','f1']
t2.columns=['DATE','TICKER','f1']
tt=t1.merge(t2,on=['TICKER','DATE'],how='inner').dropna()
tt.sort_values(['TICKER','DATE'],inplace=True)
tt.groupby('TICKER').tail(5)
tar_list=tt.DATE.tail(5).to_list()
tt=tt[tt.DATE.isin(tar_list)]
np.isclose(tt.iloc[:,2],tt.iloc[:,3]).all()
tt[~np.isclose(tt.iloc[:,2],tt.iloc[:,3])]

# 因子相关性用
def check_corr(df1,df2):
    df=df1.merge(df2,on=['DATE','TICKER'],how='inner')
    cor=df.iloc[:,2:].corr()
    return cor

ll=os.listdir(r'C:\Users\admin\Desktop\test')
for f in ll:
    print(f)
    p=r'D:\tyx\检查更新\2021.6-2025.8'
    df1=feather.read_dataframe(os.path.join(p,f.replace('_min','')))
    df2=feather.read_dataframe(os.path.join(p,f))
    print(check_corr(df1,df2))