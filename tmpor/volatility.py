import pandas as pd
import warnings

def iponew(x):
    import warnings
    warnings.filterwarnings('ignore')
    x=x.sort_values(by=['DATE']).reset_index(drop=True)
    x['newday']= [i+1 for i in range(x.shape[0])]
    return x

def cal_volatility():
    warnings.filterwarnings('ignore')
    df=pd.read_feather(r'\\zkh\O\Data_Storage\Data_Storage\daily.feather')[['DATE','TICKER','close']]
    adj_df=pd.read_feather(r'\\zkh\O\Data_Storage\Data_Storage\adj_factors.feather')
    st_df=pd.read_csv(r'\\zkh\O\Data_Storage\st.csv',encoding='gbk')
    st_df.rename(columns={'Unnamed: 0': 'DATE'}, inplace=True)
    # 股票代码是列名，将其变成一列，好merge,后面直接剔除
    st_df=st_df.melt(id_vars=['DATE'],var_name='TICKER',value_name='st')
    # 日期变成跟df格式一样的
    st_df.DATE=st_df.DATE.str.replace('-','')
    # ---------------------------------------------------
    # 剔除新股
    df=df.groupby('TICKER').apply(iponew).reset_index(drop=True)
    df=df[df['newday']>20].drop(['newday'],axis=1)
    # 合并复权因子和st数据
    df=df.merge(adj_df,on=['DATE','TICKER'],how='left')
    df=df.merge(st_df,on=['DATE','TICKER'],how='left')
    # 剔除st数据
    df.dropna(axis=0,how='any',inplace=True)
    # 计算波动率
    df.sort_values(['TICKER','DATE'],inplace=True)
    df.close=df.close*df.adj_factors
    df['pre_close']=df.groupby('TICKER')['close'].shift(1)
    df['ret']=df['close']/df['pre_close']
    df['volatility']=df.groupby('TICKER')['ret'].rolling(window=5,min_periods=5).std().values
    df=df[df.DATE>='20220101']
    # 波动率排名
    df['rank']=df.groupby('DATE')['volatility'].rank(method='min').values
    return df

if __name__=="__main__":
    df=cal_volatility()
    top100=df.groupby('DATE',as_index=False).apply(lambda x: x[x['rank']<=100])[['TICKER','DATE','rank']]
    top100.to_feather(r'C:\Users\Administrator\Desktop\top100.feather')