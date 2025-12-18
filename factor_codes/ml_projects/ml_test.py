import os
import random
import warnings
import feather
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def ml_alg():
    warnings.filterwarnings('ignore')
    # file_list=os.listdir(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子')
    # file_list.remove('RSkew_daily.feather')
    # file_list.remove('SSF_v2_half10.feather')
    # file_list.remove('VaR_5%_v2.feather')
    # file_list = random.sample(file_list, 10)
    file_list = ['有效100s买入额_roll20.feather',
                 '有效100s卖出额.feather',
                 '有效100s卖出额_roll20.feather',
                 '波动公平因子_roll20.feather',
                 '随波逐流.feather',
                 'net_trade_ratio.feather',
                 'Traction_LUD_trade_num.feather']
    bt_path=r'\\Desktop-79nue61\因子测试结果\田逸心\原因子增量回测20-21'
    start='20220101'
    end='20250901'
    pos_factors_list =[]
    neg_factors_list =[]
    res=pd.DataFrame()
    for file in tqdm(file_list):
        tmp=feather.read_dataframe(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file}')
        tmp=tmp[tmp['DATE']>='20211201'].reset_index(drop=True)
        bt=pd.read_csv(os.path.join(bt_path,file.replace('feather','csv')))
        the_col=np.setdiff1d(tmp.columns,['DATE','TICKER'])
        tmp.dropna(inplace=True)
        # tmp.sort_values(['DATE',the_col.tolist()[0]],inplace=True)
        tmp.reset_index(drop=True,inplace=True)
        if bt.tail(1).iloc[0,1]<0: #负向，取小值
            neg_factors_list.append(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file}')
        elif bt.tail(1).iloc[0,1]>0:
            tmp[the_col.tolist()[0]]*=-1
            pos_factors_list.append(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{file}')
        if res.empty:
            res=tmp
        else:
            res=res.merge(tmp,on=['DATE','TICKER'],how='inner')
        # print(res.shape)
    # print(pos_factors_list)
    # print(neg_factors_list)
    # todo: random forest------------------
    # X: res, Y:return, 得想想多久预测下一天
    # X和y的关系 是一行对应一行
    ret_df=feather.read_dataframe(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\return_因子组合用.feather')
    res=res.merge(ret_df,on=['DATE','TICKER'],how='inner')
    res['exp_ret']=res['return'].shift(-1)
    res.dropna(inplace=True)
    # todo:
    res=res[(res['DATE']>='20220101')&(res['DATE']<='20230101')]
    X=res.iloc[:,2:-2]
    y=res.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 模型训练：
    rf=RandomForestRegressor(n_estimators=100,random_state=123)
    rf.fit(X_train,y_train)
    # 模型测试：
    y_pred=rf.predict(X_test)
    # mse = mean_squared_error(y_test, y_pred)
    print(np.corrcoef(y_test, y_pred))


if __name__=='__main__':
    ml_alg()
    # if os.path.exists(r'C:\Users\admin\Desktop\codes.feather'):
    #     df=feather.read_dataframe(r'C:\Users\admin\Desktop\codes.feather')
    #     df['DATE']=df['DATE'].astype(str)
    # ret_df=feather.read_dataframe(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\return_因子组合用.feather')
    # df=df.merge(ret_df,on=['DATE','TICKER'],how='left')
    # res=df.groupby('DATE').agg({'TICKER':'size','return':'mean'}).reset_index()
    # dd=['20240206', '20240207', '20240208','20240926', '20240927', '20240930', '20241008']
    # res=res[~res['DATE'].isin(dd)]
    # print(res)
    # # feather.write_dataframe(res,r'C:\Users\admin\Desktop\res.feather')
    # res.to_csv(r'C:\Users\admin\Desktop\res.csv')
