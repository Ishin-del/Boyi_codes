# LSTM预测收益率  复盘



## processing.py

```python
import pymssql
import os
from tqdm import tqdm
import pdb
import pandas as pd
import time
import numpy as np
import torch
import random 
import warnings
warnings.filterwarnings('ignore')

# 数据预处理
# get数据
def sql_read(date_range,pool='中证1000',start_date='2021-01-04',end_date='2023-12-31',server='10.60.0.165',user='read_only',password='read_only',database='datayes'): # ,driver = 'SQL Server Native Client 11.0'
    print('Read the stock pool...')
    t1=time.time()
    # 选取上证50，2023-01-31的50支股票为标的
    # sql="""select top(1000) trddt ,ticker_symbol,[Weight] 
    # from [API_Download].[dbo].[benchcmp] 
    # where Indexname='上证50' and trddt='2023-01-31'
    # order by 'ticker_symbol'
    # """ #50只票
    
    # sql="""select top(500) TICKER_SYMBOL ticker_symbol
    #     from datayes.dbo.mkt_adjf 
    #     group by TICKER_SYMBOL """ # 500只票
    # conn=pymssql.connect(server,user,password,database)
    # conn=pyodbc.connect('DRIVER=ODBC Driver 18 for SQL Server;SERVER={};DATABASE={};UID={};PWD={}'.format(server,database,user,password))
    # if conn:
    if pool=="全A":
        # sql="""select distinct(TICKER_SYMBOL) from [datayes].[dbo].[md_sec_type] where TYPE_NAME='全部A股'"""
        # code_df=pd.read_sql(sql,conn).rename(columns={'TICKER_SYMBOL':'code'})
        # conn.close() 
        code_df=pd.read_csv('./data/all_stock.csv')
        stocks_df=code_df.code.unique().tolist()
    else:
        # sql="""select trddt ,ticker_symbol
        # from [API_Download].[dbo].[benchcmp] 
        # where Indexname='{}' and trddt between '{}' and '{}'
        # order by trddt
        # """.format(pool,start_date,end_date)
        # code_df=pd.read_sql(sql,conn)
        # conn.close() 
        # code_list=code_df.ticker_symbol.values.tolist()
        # time_pool=pd.Series(code_df.trddt.astype('str').unique())
        # stocks=pd.Series(code_df.groupby('trddt')['ticker_symbol'].apply(list).values)
        # stock_pool=pd.concat([time_pool,stocks],axis=1).rename(columns={0:'date',1:pool})
        # stocks_df=pd.merge(pd.DataFrame(date_range,columns=['date']),stock_pool,on='date',how='outer').bfill()
        code_df=pd.read_csv('./data/all_stock.csv')
        stocks_df=code_df.code.unique().tolist()[:10]
        # random.seed(123)
        # stocks_df=random.sample(stocks_df,10)
    t2=time.time()
    print("[timing:{}s]".format(t2-t1))
    return stocks_df # stock_pool #code_df#,code_list

def get_test_date(day_price,year,time_len): #,gap
    print('Getting the test date...')
    date_df=day_price
    date_df['ret']=date_df.groupby('code').apply(lambda x: (x['open'].shift(-2)-x['open'].shift(-1))/x['open'].shift(-1)).values
    # date_df['ret']=date_df.groupby('TICKER_SYMBOL').apply(lambda x:(x.CLOSE_PRICE_1.shift(-1)-x.CLOSE_PRICE_1)/x.CLOSE_PRICE_1).values
    # date_df=pd.read_csv('./data/price2016-2023.csv')
    # pdb.set_trace()
    sub_df=date_df[['date']].drop_duplicates()
    sub_df['year']=sub_df.date.str.split('-',expand=True)[0]
    tmp=sub_df[sub_df.year==year] #确定test的时间
    # 加上需要额外train的数据
    end=tmp.index.tolist()[0]
    train_set=sub_df.iloc[end-int(time_len*244):end]
    dd=pd.concat([train_set,tmp])
    date_range=dd.date.unique().tolist()#[-300:]  #[-953:] #len=1945
    # date_range=date_range[-300:]
    # date_range=date_range[-727+17:]
    # date_range=date_range[-450:]
    # pdb.set_trace()
    # if 'avg_price' not in date_df.columns:
    #     date_df['avg_price']=date_df.money/date_df.vol
    ret_df=date_df[['date','code','ret']]
    return date_range,ret_df


# 生成新数据==============================================================================================================
# 日线因子
def daily_data(day_price):
    print('Generate new features from daily data...')
    t1=time.time()
    data=day_price
    # data=pd.read_csv("./data/day_price2016-2023.csv",names=['code','date','money','open','high','low','close','vol'])
    data['avg_price']=data.money/data.vol
    # data['ret']=data.groupby('code').apply(lambda x: (x['avg_price'].shift(-1)-x['avg_price'])/x['avg_price']).values
    # data['ret']=data.groupby('code').apply(lambda x: (x['vwap'].shift(-1)-x['vwap'])/x['vwap']).values #return: 根据T+11和T+1的vwap计算
    # data['ret']=data.groupby('code').apply(lambda x:(x.close.shift(-1)-x.close)/x.close).values
    data['5_mov']=data.groupby('code').apply(lambda x: x.close.rolling(5).mean()).values
    data['10_mov']=data.groupby('code').apply(lambda x: x.close.rolling(10).mean()).values
    data['20_mov']=data.groupby('code').apply(lambda x: x.close.rolling(20).mean()).values
    t2=time.time()
    print("[timing:{}s]".format(t2-t1))
    return data

# 周度数据
def weekly_data():
    print('Generate new features from weekly data...')
    t1=time.time()
    data=pd.read_csv('./data/week_price2016-2023.csv',names=['sec_id','date','open','high','low','close','money','vol','code'])
    data['avg_price']=data.money/data.vol
    # data['ret']=data.groupby('code').apply(lambda x:(x.close.shift(-1)-x.close)/x.close).values
    data['2week_mov']=data.groupby('code').apply(lambda x: x.close.rolling(2).mean()).values
    data['4week_mov']=data.groupby('code').apply(lambda x: x.close.rolling(4).mean()).values
    t2=time.time()
    print("[timing:{}s]".format(t2-t1))    
    return data  
    
# money_flow
def money_flow_data(day_price):
    print('Generate new features from money flow data...')
    t1=time.time()
    # df1=pd.read_csv("./data/price2016-2023.csv",index_col=0).rename(columns={'TRADE_DATE':'date','TICKER_SYMBOL':'code',
    #                                                        'OPEN_PRICE_1':'open','HIGHEST_PRICE_1':'high',
    #                                                        'LOWEST_PRICE_1':'low','CLOSE_PRICE_1':'close',
    #                                                        'TURNOVER_VOL':'vol'})
    money_flow=pd.read_csv('./data/money_flow.csv',names=['code','date','inflow_s','inflow_m','inflow_x','inflow_xl','outflow_s' ,'outflow_m','outflow_l','outflow_xl'])
    data=pd.merge(day_price,money_flow,how='inner',on=['date','code'])
    # pdb.set_trace()
    # data['ret']=data.groupby('code').apply(lambda x:(x.close.shift(-1)-x.close)/x.close).values
    money_flow=data[['code','date','inflow_s','inflow_m','inflow_x','inflow_xl','outflow_s' ,'outflow_m','outflow_l','outflow_xl','money']]
    t2=time.time()
    print("[timing:{}s]".format(t2-t1))
    return money_flow

def pv_factor(day_price):
    print('Generate new features from pv factor data...')
    t1=time.time()
    # data=pd.read_csv("./data/day_price2016-2023.csv",names=['code','date','money','open','high','low','close','vol'])
    data=day_price
    data['day_ret']=data.groupby('code').apply(lambda x:(x.close.shift(-1)-x.close)/x.close).values # 日收益率
    data['open_ret']=data.groupby('code').apply(lambda x:(x.open-x.close.shift(1))/x.close.shift(1)).values # 开盘收益率
    data['vibration']=data.groupby('code').apply(lambda x: x.high/x.low-1).values # 振幅
    data['3_ret']=data.groupby('code').apply(lambda x: (x.close-x.close.shift(3))/x.close.shift(3)).values # 3日收益率
    data['5_ret']=data.groupby('code').apply(lambda x: (x.close-x.close.shift(5))/x.close.shift(5)).values # 5日收益率
    data['3_std']=data.groupby('code').apply(lambda x: x.day_ret.rolling(3).std()).values
    data['5_std']=data.groupby('code').apply(lambda x: x.day_ret.rolling(5).std()).values
    data['illiq']=data.day_ret.abs()/data.money
    # 计算换手率
    """
    equ_share从这里拉来的：
    select TICKER_SYMBOL,CHANGE_DATE,TOTAL_SHARES 
    from [DataYes].[dbo].[equ_shares_change]
    order by TICKER_SYMBOL,CHANGE_DATE
    """
    equ_share=pd.read_csv('./data/equ_share.csv',names=['code','date','share'])
    equ_share=equ_share[equ_share.code.apply(lambda x : not x.startswith('DY'))]
    equ_share.code=equ_share.code.astype('int') 
    col=data[['date','code']]
    equ_share=pd.merge(equ_share,col,on=['date','code'],how='outer')    
    equ_share['share']=equ_share.groupby('code').apply(lambda x:x.share.ffill()).values  
    data=pd.merge(data,equ_share,on=['date','code'],how='left')
    data['turn']=data.money/data.share
    # 日内波动率 计算
    """
    从这拉的：
    select trddt,ticker_symbol,factor_vol2
    from FactorRoom.dbo.Tech_ht_minute
    where trddt between '2016-01-01' and '2023-12-31'
    """
    minutely_std=pd.read_csv('./data/minutely_std.csv',names=['date','code','minutely_std'])
    data=pd.merge(data,minutely_std,on=['code','date'],how='left')
    data=data[['date','code','day_ret','open_ret','money','turn','vibration','3_ret','5_ret','3_std','5_std','illiq','minutely_std']]
    t2=time.time()
    print("[timing:{}s]".format(t2-t1)) 
    return data

def get_h5data(path,freq='30M'):
    files=os.listdir(path)
    data=pd.DataFrame()
    for f in tqdm(files):
        f_path=os.path.join(path,f)
        fs=os.listdir(f_path)
        for file in fs:
            if file.find(freq)!=-1:
                tmp_df=pd.read_hdf(os.path.join(f_path,file))
                data=pd.concat([data,tmp_df])
    return data

def get_close(df):
    close_df=df.groupby('date')['date','close'].apply(lambda x: x.iloc[-1])
    close_df['ret']=(close_df.close.shift(-1)/close_df.close-1).values
    close_df=pd.DataFrame(close_df.values).rename(columns={0:'date',1:'close',2:'ret'}).drop('close',axis=1)
    df=pd.merge(df,close_df,how='outer',on='date')
    return df


# 分钟数据
def minutely_data():
    print('Generate new features from minutely data...')
    t1=time.time()
    # sse_path='./data/minute_data/SSE/'
    # sse_data=get_h5data(sse_path)
    # szse_path='./data/minute_data/SZSE/'
    # szse_data=get_h5data(szse_path)s
    # data=pd.concat([sse_data,szse_data])
    # data=pd.read_hdf('./data/30M_price.h5')
    # data.Code=data.Code.str.split('.',expand=True)[0]
    # data.Date=data.Date.astype(str).apply(lambda x: x[:4]+'-'+x[4:6]+'-'+x[6:])
    # # data.Time=data.Time/100
    # data=data.drop('MatchItems',axis=1).drop('Interest',axis=1).rename(columns={'Code':'code',
    #                                                                        'Date':'date',
    #                                                                        'Time':'time',
    #                                                                        'Open':'open',
    #                                                                        'High':'high',
    #                                                                        'Low':'low',
    #                                                                        'Close':'close',
    #                                                                        'Volume':'vol',
    #                                                                        'Turnover':'money'})
    # data.code=data.code.astype(int)
    # """
    # 复权因子读取代码：
    # select TICKER_SYMBOL,TRADE_DATE,ACCUM_ADJ_FACTOR
    # from datayes.dbo.mkt_equd_adj
    # where TRADE_DATE between '2016-01-01' and '2023-12-29'
    # """
    # adjust_factor=pd.read_csv('./data/adjust_factor.csv',names=['code','date','adjust']).sort_values(by=['date','code'])
    # # pdb.set_trace()
    # data=pd.merge(data,adjust_factor,on=['date','code'],how='left') 
    # data.open=data.open*data.adjust
    # data.high=data.high*data.adjust
    # data.low=data.low*data.adjust
    # data.close=data.close*data.adjust
    # data.vol=data.vol/data.adjust
    """
    错误代码：
    # 算错了，不对:data['ret']=data.groupby(['code','date'])[['close']].apply(lambda x:x[-1])
    # data=pd.read_hdf('./data/minutely_price.h5')
    # close_price=data.groupby('code').apply(lambda x:x.groupby('date').apply(lambda x: x.close.iloc[-1])).reset_index().rename(columns={0:'close'})
    # close_price['ret']=close_price.groupby('code').apply(lambda x:(x.close.shift(-1)-x.close)/x.close).values 
    # ret=close_price[['code','date','ret']]
    # data=pd.merge(data,ret,on=['code','date'],how='left')  
    # data=data.sort_values(by=['date','time','code'])
    """
    # data.to_hdf('./data/minutely_price.h5',key='data')
    data=pd.read_hdf('./data/30M_price.h5')
    t2=time.time()
    print("[timing:{}s]".format(t2-t1))
    return data

def fill_series(df):
    # 写废的函数，留个思路
    time_series=[93000,100000,103000,110000,113000,133000,140000,143000,150000]
    diff=list(set(time_series).difference(set(df.time)))
    if diff==[]:
        return df
    for t in diff:
        df.loc[df.index[-1]+1]=np.NaN
        df.code,df.time=df.code.ffill(),df.time.ffill()
        df.loc[df.index[-1]+1,'time']=t
    return df

# 日内分布
def intra_return():
    print('Generate new features from intra return data...')
    t1=time.time()
    df=pd.read_hdf('./data/30M_price.h5')
    df=df.sort_values(by=['code','date','time'])
    # TODO:
    df['intraday_ret']=df.groupby('code').apply(lambda x: x.close/x.close.shift(1)-1).values
    early_ret=df.groupby(['code','date']).apply(lambda x: np.nan if len(x)<2 else x.intraday_ret.iloc[1]) #早盘30分钟收益率
    data=early_ret.reset_index().rename(columns={0:'early_ret'})
    data['open_ret']=df.groupby(['code','date']).apply(lambda x:x.intraday_ret.iloc[0]).values #开盘收益率
    data['max_ret']=df.groupby(['code','date']).apply(lambda x: np.nan if len(x)==1 else max(x.intraday_ret.iloc[1:])).values
    data['min_ret']=df.groupby(['code','date']).apply(lambda x: np.nan if len(x)==1 else min(x.intraday_ret.iloc[1:])).values
    data['avg_ret']=df.groupby(['code','date']).apply(lambda x: np.nan if len(x)==1 else np.mean(x.intraday_ret.iloc[1:])).values
    data['last_ret']=df.groupby(['code','date']).apply(lambda x : np.nan if len(x)!=9 else x.intraday_ret.iloc[-1]).values

    # ret=df.groupby(['code','date'])[['code','date','ret']].apply(lambda x: x.iloc[0])
    # df=pd.read_hdf('./data/ret.h5')
    # data=pd.merge(data,df,on=['date','code'],how='left')
    data.to_hdf('./data/intra_return.h5',key='data')
    data=pd.read_hdf('./data/intra_return.h5')
    # pdb.set_trace()
    t2=time.time()
    print("[timing:{}s]".format(t2-t1))
    return data

def get_nn_score():
    factors=os.listdir('./factors/')
    df=pd.DataFrame()
    for factor in factors:
        f_pth=os.path.join('./factors/',factor)
        df_f=pd.read_csv(f_pth).rename(columns={'factor':factor}).drop('Unnamed: 0',axis=1)
        if factors.index(factor)==0:
            df=df_f
        else:
            df=pd.merge(df,df_f,on='date')
    return df


# 标准化==============================================================================================================
def z_score(x):
    return (x-x.mean())/x.std()
# 价格数据预处理
def price_rela_processing(data,columns:list,close_col): #data=daily_data
    '''
    :param data: 要处理的df,数据表
    :param columns: 跟价格相关的columns的名称
    :return:
    '''
    df_price=data[columns]
    df_rest=data[data.columns.drop(columns).to_list()]
    # 价格数据除以最新收盘价
    # 以日度数据为例，df所有跟价格相关的数据/时间序列最后一天的close price => 所有价格相对于最新收盘价的百分比
    # 那还需要最后一天吗？(最后一天的close肯定等于1)
    df_price=df_price.astype(float)
    df_price=df_price.div(df_price.loc[:,close_col].iloc[-1],axis=0)
    df_price=df_price.apply(lambda x: z_score(x))
    data=pd.concat([df_rest,df_price],axis=1)
    return data

def turnover_processing(data,columns:list):
    # 除以序列均值做均值标准化 column_mean=data.loc[:,column].mean(),z_score已经计算过了
    # 做z_score
    for col in columns:
        data.loc[:,col] = z_score(data.loc[:,col])
    return data

# 缺失值处理==============================================================================================================
def fillna(data,m='fill_0'): 
    # print("data information:================================")
    # print(data.info())
    # 待调试
    if m=='fill':
        for c in data.columns:
            if data[c].isna().any():
                filtered_data=data[c][~np.isnan(data[c])]
                hist,bins=np.histogram(filtered_data,bins=20)
                data[c].replace(np.nan,np.mean([bins[np.argmax(hist)],bins[np.argmax(hist)+1]]),inplace=True)
        return data
    elif m=='fill_0':
        data=torch.where(torch.isnan(data),torch.full_like(data,0),data)
        data=torch.where(torch.isinf(data),torch.full_like(data,0),data)
        return data
    elif m=='drop':
        data=torch.masked_select(data,torch.isnan(data))
        return data

def batch_normalization(batch_data,factor):
    if factor=='daily_bar':
        batch_data=price_rela_processing(data=batch_data,columns=['open','high','low','close','5_mov','10_mov','20_mov','avg_price'],close_col='close')
        batch_data=turnover_processing(data=batch_data,columns=['vol','money'])
        batch_data=batch_data.drop('ret',axis=1)
        # pdb.set_trace()
    elif factor=='money_flow_factor':
        tmp_df=batch_data[['inflow_s','inflow_m','inflow_x','inflow_xl','outflow_s' ,'outflow_m','outflow_l','outflow_xl']].div(batch_data.money,axis=0)
        tmp_df=z_score(tmp_df) 
        tmp_df=pd.concat([batch_data.code,tmp_df],axis=1)
        # batch_data=tmp_df
        batch_data=pd.concat([batch_data.date,tmp_df],axis=1)
    elif factor=='weekly_bar':
        batch_data=price_rela_processing(data=batch_data,columns=['open','high','low','close','avg_price','2week_mov','4week_mov'],close_col='close')
        batch_data=turnover_processing(data=batch_data,columns=['vol','money']).drop('sec_id',axis=1)
    elif factor=='minutely_bar':
        # if batch_data.isnull().values.any():
        #     continue
        batch_data=price_rela_processing(data=batch_data,columns=['open','high','low','close'],close_col='close')
        batch_data=turnover_processing(data=batch_data,columns=['vol','money']) #'turn',
        batch_data=batch_data.sort_values(by=['date','time','code'])
    elif factor=='intra_return':
        # if batch_data.isnull().values.any():
        #     continue
        batch_data.iloc[:,2:]=z_score(batch_data.iloc[:,2:])
    elif factor=='pv_factor':
        # pdb.set_trace()
        batch_data.iloc[:,2:]=batch_data.iloc[:,2:].div(batch_data.money,axis=0)
        batch_data=batch_data.drop('money',axis=1)   
        batch_data.replace([np.inf, -np.inf], np.nan, inplace=True)   
        # batch_data.fillna(0,inplace=True)
        for col in batch_data.columns:
            if col != 'code' and col!='date':
                batch_data[col]=z_score(batch_data[col])
    return batch_data
```



## data_prepare.py

```python
import numpy as np
# import time
import torch
from processing import *
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import pdb

# 数据分割==============================================================================================================
def _data_split(dataset,n_input,n_output,y_col,train_percent):
    '''
    n_input: 滑动窗口的值的行数,30
    n_output: 预测的值的行数
    train_percent: train数据集的百分比
    y_col: 用作预测结果的列名
    price_cols:list,需要进行预处理的价格相关的列名
    vol_cols:list,需要进行预处理的成交量/额相关的列名
    '''
    x_data=dataset[dataset.columns[dataset.columns!=y_col]]
    y_data=dataset[[y_col]]
    train_size=int(len(dataset)*train_percent)
    # test_size=len(dataset)-train_size
    # X: 
    train_set=x_data.iloc[:train_size,:]
    test_set=x_data.iloc[train_size:,:]
    # Y:
    y_train=y_data.iloc[:train_size,:]
    y_test=y_data.iloc[train_size:,:] 
    # np.vstack => 相当于拼接了两个df,变成了ndarray,ndim是2
    all_data=np.vstack((train_set,test_set))
    y_set=np.vstack((y_train,y_test))[:,0] # => 格式是ndarray,把它想成一个table,[:,0]取的是所有行，第一列的数据
    # 这里y只有一列，所以y_set依然是取了整个daily_y
    X=np.empty((1,n_input,all_data.shape[1])) #all_data.shape=> (1000,6),因此shape应该是(1,30,6),ndarray
    y=np.empty((1,n_output)) #shape是(1,1)
    # pdb.set_trace()
    for i in range(all_data.shape[0] - n_input - n_output): #1000 - 30 - 1 
        #带入对数据的标准化方法--------------------
        X_sample = all_data[i:i+n_input,:] # all_data[0:30,:] => 这是按天取
        y_sample=y_set[i + n_input:i+n_input+n_output] # y_set[0+30:0+30+1], 这取的是第31
        if i==0:
            X[i]=X_sample
            y[i]=y_sample
        else:
            # 这里append相当于增加了数据维度，X：3D数据，这里增加了第一维（样本点），第二维：序列长度，第三维：特征数
            X=np.append(X,np.array([X_sample]),axis=0)
            y=np.append(y,np.array([y_sample]),axis=0)
    # 这里相当于是把X,y做一个切分，分为train和test
    # pdb.set_trace()
    train_X=X[:train_set.shape[0]-n_input,:,:] # X[0:800-30,:,:],表示730个样本点，（每个样本点序列长30，特征6）
    train_y=y[:train_set.shape[0]-n_input,:] # y[0:800-30,:],表示730个样本点，（每个样本点对应x序列30下一个的数）
    test_X=X[train_set.shape[0]-n_input:all_data.shape[0]-n_input-n_output,:,:] # X[800-30:1000-30-1,:,:]
    test_y=y[train_set.shape[0]-n_input:all_data.shape[0]-n_input-n_output,:] # y[800-30:1000-30-1,:]
    return train_X,train_y,test_X,test_y


# 根据股票分组，股票组内应用滑动分组：
def groupby_code(code_list,train_data,window=30,output=1,percent=0.9,label='ret'): 
    """
    code_list: 股票池
    train_data: 要进行切割的数据
    window: 窗口大小
    output: 预测一天or多少天
    percent: train所占数据比例 
    label: 用作label的列名
    """
    print('Data Spliting based on code...')
    t1=time.time()
    # data_size=len(train_data[train_data.code==code_list[1]].drop(['code'],axis=1))
    # new_code_list=[]
    for code in code_list:
        code_data=train_data[train_data.code==code].drop(['code'],axis=1) # 拉取同只股票
        # if len(code_data)!=data_size:
        #     continue
        # new_code_list.append(code)
        code_train_X,code_train_y,code_test_X,code_test_y=_data_split(code_data,n_input=window,n_output=output,train_percent=percent,y_col=label)
        # print(code,':',code_train_X.shape)
        if code_list.index(code)==0:
            train_X=code_train_X
            train_y=code_train_y
            test_X=code_test_X
            test_y=code_test_y
            
        else:
            train_X=np.concatenate([train_X,code_train_X],axis=0)
            train_y=np.concatenate([train_y,code_train_y],axis=0)
            test_X=np.concatenate([test_X,code_test_X],axis=0)
            test_y=np.concatenate([test_y,code_test_y],axis=0)
        
    # pdb.set_trace()
    # 转换数据类型：
    train_X=torch.from_numpy(train_X).type(torch.Tensor)
    train_y=torch.from_numpy(train_y).type(torch.Tensor)
    test_X=torch.from_numpy(test_X).type(torch.Tensor)
    test_y=torch.from_numpy(test_y).type(torch.Tensor)
    t2=time.time()
    print("[timing:{}s]".format(t2-t1))
    return  train_X,train_y,test_X,test_y #,new_code_list

def fill_df(df):
    # 此函数用缺失值补齐数据长度
    if len(df)<8:
        end_index=df.index.values[-1] #获取最后一个index的值
        start_index=df.index.values[0]
        miss_len=8-len(df.index.values)
        df=df.reindex(range(start_index,end_index+miss_len+1))
    # if len(df)<8:
    #     miss_len=8-len(df.index.values)
    #     for c in miss_len():
    #         df.loc[len(df)]=np.NaN
        df.iloc[:,6:11]=df.iloc[:,6:11].ffill()
        df.iloc[:,3:6]=df.iloc[:,3:6].fillna(0)        
    return df


def dynamic_data_prepare(train_data,percent,out,code_list,stocks_df,date_range,ret_df,stock='hs300',windows=30,factor='weekly_bar',gap=22):
    """
    date_range: 这轮train_data数据对应的 要进行预测的 时间序列（单位：天） 
    """
    print('data prepareing for the model ...')
    t1=time.time()
    # pdb.set_trace()
    # print('number of training stock is {}'.format(len(code_list[:500])))
    if 'time' in train_data.columns:
        # 过滤掉9：30 数据
        # pdb.set_trace()
        train_data=train_data[~train_data['time'].isin([93000])]
    x_train_dataloader,y_train_dataloader,x_test_dataloader,y_test_dataloader=[],[],[],[]
    # all_X,all_y,all_y_date=[],[],[]
    y_date_test,y_date_train=[],[]
    for day in tqdm(range(windows,len(date_range))): 
        # windows内数据是特征数据，1是label，也就是要预测的数据。因此，要以第31天的股票池往上拉
        if stock=='全A':
            # code_list_str=code_list
            # code_list_int=list(map(lambda x:int(x),code_list_str)) 
            #==============
            code_list_int=code_list
            # pdb.set_trace()
        # elif stock=='hs300':  
        else:      
        # 拿到当天股票池
            # pdb.set_trace()
            # code_list_str=stocks_df[stocks_df.date==date_range[day]].iloc[0,1]
            # # pdb.set_trace()
            # code_list_int=list(map(lambda x:int(x),code_list_str)) #转成int格式，后面跟train_data做交集
            #==============
            code_list_int=code_list #[:10]
        # 向上拿30天
        # train_data是截取的固定时长的数据 
        # 这里已经把数据拉成30和1了
        train_date=date_range[day-windows:day] #[day-windows:day+1] # 最后加的这1行的ret属于这组x对应的y数据
        # pdb.set_trace()
        test_date=date_range[day]
        batch_data=train_data[train_data.date.isin(train_date)]
        batch_data=batch_data[batch_data.code.isin(code_list_int)]
        # pdb.set_trace()
        # 按股票代码排序排好
        batch_data=batch_data.sort_values(by='code')
        # batch_data是从train_data处截取的选中股票和窗口时间的数据
        code_set=batch_data.code.unique().tolist()
        # num_stock=len(code_set)
        # batch_data=batch_data.set_index("date")
        # 中性化
        batch_data=batch_normalization(batch_data,factor)
        # 根据股票分组：
        # TODO:到这里，batch_data（是个df）的shape应该是 （windows的长*股票数）*列数
        X_set,y_set,y_date=[],[],[]
        # pdb.set_trace()
        for code in code_set:
            # pdb.set_trace()           
            if 'time' in batch_data.columns:
                code_data=batch_data[batch_data.code==code].sort_values(by=['date','time']) #.reset_index()
            else:
                code_data=batch_data[batch_data.code==code].sort_values(by='date') #.reset_index()
            # y_date.append(code_data.date.iloc[-out])
            # pdb.set_trace()
            if factor=='weekly_bar':
                pass
            elif set(train_date).difference(set(code_data.date.unique().tolist()))!=set():
                continue
                # date_set=list(set(train_date).difference(set(code_data.date.unique().tolist())))
                # for i in range(len(date_set)):
                #     code_data.loc[len(code_data)]=np.nan
                #     code_data.loc[len(code_data),'date']=date_set[i]
            # pdb.set_trace()
            # X=pd.Series()
            if factor!='minutely_bar':
                X=code_data[code_data.columns.drop('code').drop('date').tolist()] #.iloc[:windows,:] #.drop('ret')
            elif len(code_data) < len(train_date)*8: # 说明分钟级数据，数据不完整
                # tmp_array=code_data.groupby('date').apply(lambda x: x.reindex(range(x.index.values[0],x.index.values[-1]+(8-len(x))+1)))
                continue
                # pdb.set_trace()
                # tmp_array=code_data.groupby('date').apply(fill_df).values
                # code_data=pd.DataFrame(tmp_array,columns=code_data.columns)
                # X=code_data[code_data.columns.drop('code').drop('date').drop('time').tolist()] #.iloc[:windows*8,:]
            else: # 分钟级数据，且数据完整
                X=code_data[code_data.columns.drop('code').drop('date').drop('time').tolist()] #.iloc[:windows*8,:] #.drop('ret')
            # pdb.set_trace()
            # y=code_data[['ret']]
            code_ret=ret_df[ret_df.code==code]
            y=pd.Series(code_ret[code_ret.date==test_date].ret.tolist())
            if not np.isfinite(X).values.all():
                continue
            if not np.isfinite(y).values.all():
                continue
            # if X.isnull().values.any() or np.isinf(X).values.any():  #and factor!='intra_return':
            #     continue
            # elif y.isnull().values.any() or np.isinf(y).values.any():  #and factor!='intra_return': 
            #     continue
            # y_date.append(code_data.date.iloc[-out])
            y_date.append([test_date,code])
            # pdb.set_trace()
            if factor=='minutely_bar':
                # try:
                X=np.array(X).reshape(1,8*windows,len(X.columns))
                # except ValueError:
                #     pdb.set_trace()
            elif factor == 'weekly_bar':
                # try:
                tmp_window=len(X)
                X=np.array(X).reshape(1,tmp_window,len(X.columns))
                # except ValueError:
                #     pdb.set_trace()
                #     print("len(x)={}, data length is not match, drop it.".format(len(X)))
                #     continue
            else:  
                while len(X)<windows:
                    X=pd.concat([X,pd.Series(0)],axis=0).drop([0],axis=1)
                X=np.array(X).reshape(1,windows,len(X.columns)) 

            # y=code_data[['ret']] #.iloc[-1,:]
            try:
                y=np.array(y)[-out].reshape(1,1)
            except IndexError:
                continue
            # try:
            if code_set.index(code)==0: # or X_set==[]
                X_set=X
                y_set=y
            else:
                X_set=np.concatenate([X_set,X],axis=0)
                y_set=np.concatenate([y_set,y],axis=0)
                # 这里1天的数据完成，也就是一个mini-batch完成
            # except ValueError:
            #     continue
        # pdb.set_trace()
        if np.array(X_set).size==0:
            print('{} to {}, all stocks with no valid data'.format(train_date[0],train_date[-1]))
            continue
        # if np.isnan(X_set).any() or np.isinf(X).any():
        #     pdb.set_trace()
        # elif np.isnan(y_set).any() or np.isinf(y).any():
        #     pdb.set_trace()
        if factor=='minutely_bar' or factor=='intra_return':
            # try:
            X_set=X_set.astype(float)
            y_set=y_set.astype(float)
            # except AttributeError:
            #     continue
        # try:
        X_set=torch.from_numpy(X_set).type(torch.Tensor)
        y_set=torch.from_numpy(y_set).type(torch.Tensor)
        # except TypeError:
        #     continue
        # TODO:
        
        # time_list=train_data.date.unique().tolist() # 拉取这一轮数据的时间序列
        # if train_date[-1]>=date_range[int(len(date_range)*percent)-1]: #查找要做test的时间，然后这往后的append进test
        if train_date[-1]>=date_range[-gap-1]:
        # 注意这个时间要根据gap的长度变化而变化
            x_test_dataloader.append(X_set)
            y_test_dataloader.append(y_set)
            y_date_test.append(y_date) #test_date
        else:
            x_train_dataloader.append(X_set)
            y_train_dataloader.append(y_set)
            y_date_train.append(y_date) #test_date
    #     all_X.append(X_set)
    #     all_y.append(y_set)
    #     all_y_date.append(y_date)
    #     # pdb.set_trace()
    # # 分train和test
    # x_train_dataloader=all_X[:int(len(all_X)*percent)]
    # x_test_dataloader=all_X[int(len(all_X)*percent):]

    # y_train_dataloader,y_date_train=all_y[:int(len(all_X)*percent)],all_y_date[:int(len(all_X)*percent)]
    # y_test_dataloader,y_date_test=all_y[int(len(all_X)*percent):],all_y_date[int(len(all_X)*percent):]
    t2=time.time()
    print("[timing:{}s, {}min]".format(t2-t1,(t2-t1)/60))

    return x_train_dataloader,y_train_dataloader,x_test_dataloader,y_test_dataloader,y_date_train,y_date_test

```





## lstm_model.py

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from torch.nn.parallel import DistributedDataParallel 
from torch.autograd import Variable
from pytorchtools import EarlyStopping
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from processing import *
import warnings
warnings.filterwarnings('ignore')
import pdb


# __all__=['training']
# LSTM 模型==============================================================================================================
# input: 滑动窗口构建数据集
# batch: t日所有股票
class AttentionLayer(nn.Module):
    def __init__(self,hidden_size):
        super(AttentionLayer,self).__init__()
        self.hidden_size=hidden_size
        # self.attention_weight=nn.Linear(hidden_size,1,bias=False)
        # self.softmax=nn.Softmax(dim=1)
        self.linear_q = nn.Linear(hidden_size, hidden_size)
        self.linear_k = nn.Linear(hidden_size, hidden_size)
        self.linear_v = nn.Linear(hidden_size, hidden_size)

    def forward(self,lstm_output):
        # attention_scores=self.attention_weight(lstm_output)
        # attention_weights=self.softmax(attention_scores)
        # context_vector=torch.sum(attention_weights*lstm_output,dim=1,keepdim=True)
        q = self.linear_q(lstm_output)
        k = self.linear_k(lstm_output)
        v = self.linear_v(lstm_output)
        attention_scores=torch.matmul(q,k.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.hidden_size,dtype=torch.float32))
        attention_weights = torch.softmax(attention_scores,dim=-1)
        context_vector=torch.matmul(attention_weights,v)
        return context_vector


class LSTM_model(nn.Module):
    def __init__(self,input_dim,hidden_dim1,hidden_dim2,hidden_dim3,hidden_dim4,num_layers,output_dim,dropout):
        '''
        input_dim: 输入维度，即特征数量
        hidden_dim1:第一层隐层的神经元
        hidden_dim2:第二层隐层的神经元
        num_layers: 神经网络隐层几层
        output_dim: 输出维度
        '''
        super(LSTM_model,self).__init__()
        self.hidden_dim1=hidden_dim1
        self.hidden_dim2=hidden_dim2
        self.hidden_dim3=hidden_dim3
        self.hidden_dim4=hidden_dim4
        self.num_layers=num_layers

        # 第一层lstms
        self.lstm1=nn.LSTM(input_dim,hidden_dim1,num_layers,batch_first=True,dropout=dropout)
        # 第二层lstm
        self.lstm2=nn.LSTM(hidden_dim1,hidden_dim2,num_layers,batch_first=True,dropout=dropout)
        # self.attention=nn.MultiheadAttention(hidden_dim1,num_heads=1)
        # 第三层lstm
        self.lstm3=nn.LSTM(hidden_dim2,hidden_dim3,num_layers,batch_first=True,dropout=dropout)
        # 第四层lstm
        self.lstm4=nn.LSTM(hidden_dim3,hidden_dim4,num_layers,batch_first=True,dropout=dropout)
        # attention
        self.attention=AttentionLayer(hidden_dim4)
        self.fc=nn.Linear(hidden_dim4,output_dim)
        # self.fc=nn.Linear(hidden_dim1,output_dim)
    
    def forward(self,x):
        # 第一层lstm
        h0_1 = nn.init.xavier_normal(torch.empty(self.num_layers,x.shape[0],self.hidden_dim1)).requires_grad_().to(x.device)
        # c0_1 = torch.zeros(self.num_layers,x.shape[0],self.hidden_dim1).requires_grad_().to(x.device)
        c0_1 = nn.init.xavier_normal(torch.empty(self.num_layers,x.shape[0],self.hidden_dim1)).requires_grad_().to(x.device)
        out1, _ =self.lstm1(x,(h0_1.detach(),c0_1.detach()))

        # 第二层lstm
        h0_2 = nn.init.xavier_normal(torch.empty(self.num_layers,x.shape[0],self.hidden_dim2)).requires_grad_().to(x.device)
        c0_2 = nn.init.xavier_normal(torch.empty(self.num_layers,x.shape[0],self.hidden_dim2)).requires_grad_().to(x.device)
        out2, _ =self.lstm2(out1,(h0_2.detach(),c0_2.detach()))
        # out=self.fc(out2[:,-1,:])

        # 第三层lstm
        h0_3 = nn.init.xavier_normal(torch.empty(self.num_layers,x.shape[0],self.hidden_dim3)).requires_grad_().to(x.device)
        c0_3 = nn.init.xavier_normal(torch.empty(self.num_layers,x.shape[0],self.hidden_dim3)).requires_grad_().to(x.device)
        out3, _ =self.lstm3(out2,(h0_3.detach(),c0_3.detach()))

        # 第四层lstm
        h0_4 = nn.init.xavier_normal(torch.empty(self.num_layers,x.shape[0],self.hidden_dim4)).requires_grad_().to(x.device)
        c0_4 = nn.init.xavier_normal(torch.empty(self.num_layers,x.shape[0],self.hidden_dim4)).requires_grad_().to(x.device)
        out4, _ =self.lstm4(out3,(h0_4.detach(),c0_4.detach()))

        # attention
        # context_vector=self.attention(out1)
        context_vector=self.attention(out4)
        out =self.fc(context_vector[:,-1,:])
        # out =self.fc(context_vector.squeeze(1))
        return out
  
# 自写loss函数
def pearson_corr(y_pred,y):
    mean_y=torch.mean(y)
    mean_y_pred=torch.mean(y_pred)
    cov=torch.mean((y-mean_y)*(y_pred-mean_y_pred))
    std_y=torch.std(y)
    std_y_pred=torch.std(y_pred)
    corr=cov/(std_y*std_y_pred)
    return corr

class PearsonCorr(nn.Module):
    def __init__(self):
        super(PearsonCorr,self).__init__()
    
    def forward(self,pred,label):
        pred_var=Variable(pred,requires_grad=True)
        label_var=Variable(label,requires_grad=True)
        corr=pearson_corr(pred_var,label_var)
        # corr=torch.corrcoef(pred,label)
        loss = 1-corr
        return loss 

# data_spilt生成均值，通过dataloader对同一日期所有股票(面板数据)进行price和vol相关数据的归一化
def normal_batch(data_X,data_y,num_stock,window): # ,label='train'
    print('Spliting & normalizing the data...')
    t1=time.time()
    # pdb.set_trace()
    data_X=fillna(data_X)
    data_y=fillna(data_y)
    data_chunk_X=torch.chunk(data_X,num_stock,dim=0) 
    data_chunk_y=torch.chunk(data_y,num_stock,dim=0) 
    reformat_data=torch.stack(data_chunk_X,dim=1) #shape:[187, 10, 30, 9]
    reformat_y=torch.stack(data_chunk_y,dim=1) 
    batchs_X=[]
    batchs_y=[]
    # y的归一化 调用scaler
    scaler=StandardScaler()
    for i in range(len(reformat_data)):
        sub_df=pd.DataFrame(np.reshape(reformat_data[i],(window*num_stock,9)))
        sub_y=reformat_y[i]
        # 做归一化处理
        # 这里知道是0，1，2，3，6，7，8是价格相关列，4，5是vol相关列，3是close
        sub_df=price_rela_processing(data=sub_df,columns=[0,1,2,3,6,7,8],close_col=3)
        sub_df=turnover_processing(data=sub_df,columns=[4,5])
        sub_array=np.array(sub_df).reshape(num_stock,window,9)
        batchs_X.append(sub_array)
        # y也得做归一化
        # 不对，有问题的！！待改:
        # if label=='train':
        #     sub_y=scaler.fit_transform(sub_y.cpu())
        # else:
        #     pdb.set_trace()
        #     sub_y=scaler.transform(sub_y.cpu())
        batchs_y.append(sub_y)
    t2=time.time()
    print("[timing:{}s]".format(t2-t1))
    return batchs_X,batchs_y,scaler

# 模型训练==============================================================================================================
def training(x_train,y_train,x_test,y_test,input_dim,
             epoch,hidden_dim1,hidden_dim2,hidden_dim3,hidden_dim4,num_layers,output_dim,num_stock,
             window,lr,batch_size,dropout,
             shuffle,patience,delta,
             batch_norm, mode, local_rank,
             id,
             loss='mse'):
    print('Training the lstm model...')
    """
    根据模型的定义:输入形状为(batch_size,len,feature)
    增加dataloader之后,每个batch形状为(64,len,feature)
    """
    # 单机多卡！！！！初始化 ========================================================================
    # torch.cuda.set_device(local_rank)
    # torch.distributed.init_process_group(backend='nccl',init_method='env://') #,world_size=4,rank=local_rank
    
    # =============================================================================================
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu") #"cuda:{}".format(id) 
    print(device, 'is using.')
    # pdb.set_trace()
    if mode=='dynamic':
        x_train_dataloader,y_train_dataloader,x_test_dataloader,y_test_dataloader=x_train,y_train,x_test,y_test
    elif batch_norm:
        # 自写dataloader
        x_train_dataloader,y_train_dataloader,scaler=normal_batch(x_train,y_train,num_stock,window)
        x_test_dataloader,y_test_dataloader,scaler=normal_batch(x_test,y_test,num_stock,window)
    else:
        # 加dataloader
        x_train_dataloader=DataLoader(x_train,batch_size,shuffle)
        y_train_dataloader=DataLoader(y_train,batch_size,shuffle)       
        x_test_dataloader=DataLoader(x_test,batch_size,shuffle) 
        y_test_dataloader=DataLoader(y_test,batch_size,shuffle)

    model=LSTM_model(input_dim,hidden_dim1,hidden_dim2,hidden_dim3,hidden_dim4,num_layers,output_dim,dropout).to(device) 
    # -----------------------------------------------------------------------------------
    # model=DistributedDataParallel(model)
    # model=torch.nn.DataParallel(model,device_ids=[id])
    # ---------------------------------------------
    # pdb.set_trace()
    if loss=='mse':
        criterion = torch.nn.MSELoss() #之后要换掉
    elif loss=='corr':
        criterion = PearsonCorr()
    optimiser = torch.optim.Adam(model.parameters(),lr) #para的个数=隐层数量*4
    # 加early stopping
    early_stopping=EarlyStopping(patience=patience,delta=delta,verbose=True)
    for i in range(epoch):
        # Train
        model.train()
        train_pred=[]
        train_total_loss=0
        for x,y in zip(x_train_dataloader,y_train_dataloader):
            # pdb.set_trace()
            # if type(x)==torch.Tensor:
            #     x=fillna(x).to(device)
            # else:
            #     x=fillna(torch.from_numpy(x).float()).to(device) # 因为在normal_batch这里为了归一化转换了格式
            # # y=fillna(torch.from_numpy(y).float()).to(device)
            # y=fillna(y).to(device)
            x,y=x.to(device),y.to(device)
            # pdb.set_trace()
            #print(x.shape,y.shape)
            y_pred=model(x)
            # print('loss for batches:',sum(torch.pow((y_pred-y),2))/len(y))
            # pdb.set_trace()
            loss=criterion(y_pred,y)
            # print(loss.item())

            # if loss.item() == torch.exp(torch.tensor([100])).item():
            #     print('training')
            #     # pdb.set_trace()
            #     for name,param in model.named_parameters():
            #         print(f'{name}:{param.data}')
            #     model.parameters
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            train_total_loss+=loss.item()
            # y_pred_reform=scaler.inverse_transform(y_pred.cpu().detach().numpy())
            # train_pred.append(y_pred_reform.tolist())
            train_pred.append(y_pred.tolist())
        # print('x:')
        # print(x)
        # print('train y_true')
        # print(y)
        # print('train y_pred')
        # print(y_pred)
        # print('loss')
        # print(train_total_loss/len(x_train_dataloader))
        # pdb.set_trace()
        train_loss=train_total_loss/len(x_train_dataloader)
        # Test
        test_total_loss=0
        test_pred=[]
        with torch.no_grad():
            model.eval()
            # x_test=fillna(x_test).to(device)
            # y_test=fillna(y_test).to(device)
            # y_pred=model(x_test.to(device))
            # test_loss=criterion(y_pred,y_test.to(device))
            # test_pred.append(y_pred.cpu().tolist())
            # pdb.set_trace()
            for x,y in zip(x_test_dataloader,y_test_dataloader): #,y_date_test
                # print(d)
                #11
                # if type(x)==torch.Tensor:
                #     x=fillna(x).to(device)
                # else:
                #     x=fillna(torch.from_numpy(x).float()).to(device)
                # # y=fillna(torch.from_numpy(y).float()).to(device)
                # y=fillna(y).to(device)
                x,y=x.to(device),y.to(device)
                y_pred=model(x)
                loss=criterion(y_pred,y)
                test_total_loss+=loss.item()
                # y_pred_reform=scaler.inverse_transform(y_pred.cpu().detach().numpy())
                # test_pred.append(y_pred_reform.tolist())
                test_pred.append(y_pred.tolist())
                # pdb.set_trace()
            # print('y_pred')
            # print(y_pred)
            # pdb.set_trace()
            # print('y_reform')
            # print(y_pred_reform)
        test_loss=test_total_loss/len(x_test_dataloader)
        # print(f'Average loss :{total_loss/len(x_train_dataloader)}')
        # print('x_test')
        # print(x_test)
        print("Epoch ",i," Average Train loss:%.5f"%train_loss,f" Average Test loss :%.5f"%test_loss)
        early_stopping(loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return model,train_pred,test_pred #,scaler

# 下面是写废的代码，没有调用以下函数
# 模型表现==============================================================================================================
def eval(model_trained,train_y_pred,test_x,test_y,train_y,scaler):
    # 带入test_x
    test_x=fillna(test_x).cuda()
    test_y_pred=model_trained(test_x)
    test_y=fillna(test_y)
    train_y=fillna(train_y).cuda()
    train_y_pred=fillna(train_y_pred)
    # TODO:反向转换回原始的数据尺度
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())
    train_score=math.sqrt(mean_squared_error(train_y[:,0].cpu().detach().numpy(),train_y_pred[:,0].cpu().detach().numpy()))
    test_score=math.sqrt(mean_squared_error(test_y[:,0].cpu().detach().numpy(),test_y_pred[:,0].cpu().detach().numpy()))
    print('Train score: %.5f RMSE' % train_score) 
    print('Test score: %.5f RMSE'% test_score)
    return train_score,test_score,train_y_pred,test_y_pred

# 预测股价，得到ic值 ==============================================================================================================
def cal_factor_ic(pred,value):
    factor=z_score(pd.DataFrame(pred.detach().numpy()))
    ic=factor.iloc[:,0].corr(pd.DataFrame(value[:,0].detach().numpy()).iloc[:,0])
    return factor,ic

def get_factor(model_trained,train_y_pred,test_X,test_y,train_y,scaler):
    _,_,train_y_pred,test_y_pred=eval(model_trained,train_y_pred,test_X,test_y,train_y,scaler)
    factor_train,ic_train=cal_factor_ic(train_y_pred,train_y)
    factor_test,ic_test=cal_factor_ic(test_y_pred,test_y)
    print("train: IC = %.5f" % ic_train)
    print("test: IC = %.5f" % ic_test)
    # model_trained=model_trained(test_data)
    return factor_train,factor_test

```



## factor_ic.py

```python
import pandas as pd
from processing import *
import pdb
import random

def factor_ic(date,pred,true,baseline=False):
    date=date.rename(columns={0:'date',1:'code'}).reset_index().drop('index',axis=1)
    # if 'index' in date.columns:
    #     date=date.drop('index',axis=1)
    tmp=pd.concat([date,pred],axis=1).rename(columns={0:'pred'})
    res=pd.concat([tmp,true],axis=1).rename(columns={0:'true'})
    res['factor']=z_score(res.pred)
    day_ic=res.groupby('date')[['true','factor']].apply(lambda x: x.true.corr(x.factor)).to_frame().reset_index().rename(columns={0:'test_daily_ic'})
    # split_date=day_ic.date.str.split('-',expand=True)
    # ic_df=day_ic.copy()
    # ic_df['year'],ic_df['month'],ic_df['day']=split_date[0],split_date[1],split_date[2]
    # ic_df['year_month']=ic_df['year']+'-'+ic_df['month']
    # ic_year=ic_df.groupby('year')['day_ic'].mean()
    # month_ic=ic_df.groupby('year_month')['day_ic'].mean()
    if baseline==True:
        print('===============================Baseline===============================')
        # pdb.set_trace()
        res['baseline2']=[random.random() for i in range(len(res))]
        res['yesterday_true']=res.true.shift(1)
        # day_ic
        day_ic_base=res.groupby('date')[['true','yesterday_true']].apply(lambda x: x.true.corr(x.yesterday_true)).to_frame().reset_index().rename(columns={0:'baseline1_daily_ic'})
        day_ic_base2=res.groupby('date')[['true','baseline2']].apply(lambda x: x.true.corr(x.baseline2)).to_frame().reset_index().rename(columns={0:'baseline2_daily_ic'})
        # print('day ic -----------------')
        # print(day_ic_base)
        # date_base=day_ic_base.date.str.split('-',expand=True)
        # ic_base=day_ic_base.copy()
        # ic_base['year'],ic_base['month'],ic_base['day']=date_base[0],date_base[1],date_base[2]
        # ic_base['year_month']=ic_base['year']+'-'+ic_base['month']
        # year_ic
        # ic_year_base=ic_base.groupby('year')['day_ic'].mean()
        # print('year ic -----------------')
        # print(ic_year_base)
        # month_ic_base=ic_base.groupby('year_month')['day_ic'].mean()
        # print('month ic -----------------')
        # print(month_ic_base)
    else:
        ic_year_base=''
        month_ic_base=''

    return res[['date','code','factor','true']],day_ic,day_ic_base,day_ic_base2 #ic_year,month_ic,ic_year_base,month_ic_base,

def split(series):
    size=int(len(series)*0.9)
    d=[]
    # d1=[]
    for i in range(len(series)-30-1):
        # x_date=series[i:i+30].tolist()
        y_date=series[i+30:i+30+1].tolist()
        d.append(y_date)
        # d1.append(x_date)
    y_train=d[:size-30]
    y_test=d[size-30:len(series)-30-1:]
    # x_train=d1[:size-30]
    # x_test=d1[size-30:len(series)-30-1:]
    return y_train,y_test # ,x_train,x_test

def flat_list(list_series):
    res=[]
    for i in list_series:
        if isinstance(i,torch.Tensor):
            i=i.tolist()
        if isinstance(i,list):
            res.extend(flat_list(i))
        else:
            res.append(i)
    return res

def get_date_df(y_date_test):
    y_test_date=pd.DataFrame()
    for i in range(len(y_date_test)):
        if i==0:
            y_test_date=pd.DataFrame(y_date_test[i])
        else:
            # y_test_date=y_test_date.append(pd.DataFrame(y_date_test[i]))
            y_test_date=pd.concat([y_test_date,pd.DataFrame(y_date_test[i])],axis=0)
    return y_test_date
```



## main.py

```python
import pandas as pd
import re
from processing import *
from neutralize import *
from data_prepare import *
from lstm_model import *
from factor_ic import *
from config import *
# import pymssql
from sklearn.preprocessing import StandardScaler
# import pyodbc
# import faulthandler
import warnings
warnings.filterwarnings('ignore')

def normalization(array,scaler,label="train"):
    array=np.nan_to_num(array,0)
    # if 
    for i in range(len(array)):
        if array[i].shape[0]==1:
            array[i]=array[i].reshape(-1,1)
        if label=="train":
            array[i]=scaler.fit_transform(array[i])
        elif label=="test":
            array[i]=scaler.transform(array[i])
    return array,scaler
def set_seed(seed):
    torch.manual_seed(seed) #cpu种子
    torch.cuda.manual_seed_all(seed) #gpu种子

def static_training(opt):
    print('static training:')
    print('============================window={},num_stock=500,shuffle=False============================'.format(opt.window))
    # 读取2022-2023年所有股票的高开低收成交量
    data=pd.read_csv("price2016-2023.csv",index_col=0).rename(columns={'TRADE_DATE':'date','TICKER_SYMBOL':'code',
                                                                       'OPEN_PRICE_1':'open','HIGHEST_PRICE_1':'high',
                                                                       'LOWEST_PRICE_1':'low','CLOSE_PRICE_1':'close',
                                                                       'TURNOVER_VOL':'vol'})
    # 根据2023-01-31的股票作为标的
    # pdb.set_trace()
    code_list_str=sql_read('2023-01-31') 
    # code_list_str=['600010', '600028', '600030', '600031', '600036', '600048', '600104', '600111', '600196', '600276']
    code_list1=list(map(lambda x:int(x),code_list_str)) #转成int了，因为读到的data里的code就是int格式   
    # code_list1=[600010, 600028, 600030, 600031, 600036, 600048, 600104, 600111, 600196, 600276]
    # num_stock=len(code_list)
    # 上面两个算交集(相当于get_set_code的工作)
    data=data[data.code.isin(code_list1)]  
    # pdb.set_trace()
    # 读入数据，生成新特征和labels
    date,data=daily_data(data)
    # 中性化
    # size_factor,stocklist_df=neutrlize_factors(date,'2022-01-01','2023-12-30',code_list_str) #'2017-12-29'
    # data=mkt_neutral(data,size_factor,stocklist_df)
    # data=data.drop('size',axis=1)
    
    # train_data=data[data.index<=5995027].set_index("date").drop(['year'],axis=1)
    # train_data=data[data.year==2022].set_index("date").drop(['year'],axis=1)
    try:
        train_data=data.set_index("date").drop(['year'],axis=1)
    except KeyError:
        train_data=data.set_index("date")
    # pdb.set_trace()
    code_list=list(set(train_data.code.values))
    num_stock=len(set(train_data.code.values)) 
    
    # pdb.set_trace()
    # ==================================================================
    # 拿出ic的时间序列
    tt=train_data.reset_index()[['date','code']]
    # pdb.set_trace()
    y_date_train={}
    y_date_test={}
    # x_date_train={}
    # x_date_test={}
    for code in code_list:
        # pdb.set_trace()
        y_tr,y_test=split(tt[tt.code==code]['date']) #,x_tr,x_test
        y_date_train[code]=y_tr
        y_date_test[code]=y_test
        # x_date_train[code]=x_tr
        # x_date_test[code]=x_test
    # ==================================================================
    # pdb.set_trace()
    train_X,train_y,test_X,test_y,code_list=groupby_code(code_list,train_data,window=opt.window)
    # pdb.set_trace()
    num_stock=len(code_list)
    # pdb.set_trace()
    # 归一化版本1.0
    # scaler=StandardScaler()
    # # 数据喂入模型，开始训练
    # train_X,scaler=normalization(train_X,scaler)
    # # train_y,scaler=normalization(train_y,scaler)
    # test_X,scaler=normalization(test_X,scaler,label='test')
    # # test_y,scaler=normalization(test_y,scaler,label='test')
    # train_X=torch.from_numpy(train_X).type(torch.Tensor)
    # test_X=torch.from_numpy(test_X).type(torch.Tensor)
    
    model_trained,train_pred,test_pred=training(train_X,train_y,test_X,test_y,input_dim=9, 
                                                        epoch=opt.epoch,
                                                        hidden_dim1=opt.hidden_dim1,
                                                        hidden_dim2=opt.hidden_dim2, 
                                                        num_layers=opt.num_layers,
                                                        output_dim=opt.output_dim,
                                                        num_stock=num_stock,
                                                        window=opt.window,
                                                        lr=opt.lr,
                                                        batch_size=opt.batch_size,
                                                        dropout=opt.dropout,
                                                        shuffle=opt.shuffle,
                                                        patience=opt.patience,
                                                        delta=opt.delta,
                                                        batch_norm=opt.batch_norm)
                                                
    # pdb.set_trace()
    y_train_date=pd.Series(flat_list(y_date_train[code_list[1]])*num_stock) # date*10,10=batch_size
    y_test_date=pd.Series(flat_list(y_date_test[code_list[1]])*num_stock)
    train_pred=pd.Series(flat_list(train_pred)) #len==1870
    test_pred=pd.Series(flat_list(test_pred))
    train_y=pd.Series(flat_list(train_y.tolist()))
    test_y=pd.Series(flat_list(test_y.tolist()))
    # predict里的值和index都跟跟train_y里的值和index对应着的
    # pdb.set_trace()
    # train:
    _,train_year_ic,train_month_ic,train_day_ic=factor_ic(y_train_date,train_pred,train_y,baseline=True)
    print("===============================Training data===============================")
    print("2022-2023 year ic -----------------")
    print(train_year_ic)
    print("2022-2023 month ic -----------------")
    print(train_month_ic)
    print("2022-2023 daily ic -----------------")
    print(train_day_ic)
    # test:
    
    test_factor,test_year_ic,test_month_ic,test_day_ic=factor_ic(y_test_date,test_pred,test_y) #,baseline=True
    print("===============================Testing data===============================")
    print("2022-2023 year ic -----------------")
    print(test_year_ic)
    print("2022-2023 month ic -----------------")
    print(test_month_ic)
    print("2022-2023 daily ic -----------------")
    print(test_day_ic)


def get_data_para(factor,day_price):
    if factor=='daily_bar': #v
        data=daily_data(day_price)
        # data=data.drop('ret',axis=1)
        input_dim=10
        window=30
    elif factor=='money_flow_factor': #v
        # pdb.set_trace()
        data=money_flow_data(day_price)
        # data=data.drop('ret',axis=1)
        input_dim=8
        window=30
    elif factor=='weekly_bar': #v
        data=weekly_data()
        # data=data.drop('ret',axis=1)
        # pdb.set_trace()
        input_dim=9
        window=30
    elif factor=='minutely_bar':
        data=minutely_data().drop('adjust',axis=1)
        # data=data.drop('ret',axis=1)
        # 参数
        # opt.patience=5
        # opt.num_layers=10
        # opt.lr=0.0001
        # opt.hidden_dim1=256
        # opt.hidden_dim2=128
        # opt.percent=0.8
        input_dim=6
        window=5

    elif factor=='intra_return':
        data=intra_return().sort_values(by=['code','date'])
        data=data.drop('ret',axis=1)
        input_dim=6
        window=5
    elif factor=='pv_factor':
        data=pv_factor(day_price).sort_values(by=['code','date'])
        # data=data.drop('ret',axis=1)
        input_dim=10
        window=30
    return data,input_dim,window

def dynamic_training(opt,factor,day_price,date_range,ret_df,stocks_df):
    """
    time_len: 固定时长，默认在1年数据上训练
    windows: 滑动窗口大小
    out: 模型预测1个值
    gap=30:每隔多久训一次，单位是天
    """
    print('dynamic training:')
    data,input_dim,window=get_data_para(factor,day_price)
    # code_list=data.code.unique().tolist() #len=5571
    code_list=stocks_df
    # TODO:groupby起来，得到长度，用长度截取？？试试
    res_df=pd.DataFrame() # 最终ic结果的df
    factor_df=pd.DataFrame() # 最终因子结果的df
    # pdb.set_trace()
    for i in range(0,int(len(date_range)-244*opt.time_len-opt.gap),opt.gap): # 假设一年的交易日244天,一个月设为22天交易日
        sub_range=date_range[i:i+int(244*opt.time_len)+opt.gap] # 这里是，理想下：training是一年(225)，test是一个月（22）
        print('~~~~~~~~~~~~~~ data from {} to {} ~~~~~~~~~~~~~~'.format(sub_range[0],sub_range[-1]))
        train_data=data[data.date.isin(sub_range)]
        # train_data=train_data.iloc[:2000,:]
        # train_data
        # pdb.set_trace()
        x_train_dataloader,y_train_dataloader,x_test_dataloader,y_test_dataloader,y_date_train,y_date_test\
            =dynamic_data_prepare(train_data,opt.percent,opt.output,code_list,stocks_df,date_range=sub_range,ret_df=ret_df,windows=window,factor=factor,stock=opt.pool,gap=opt.gap)
        # pdb.set_trace()
        _,train_pred,test_pred=training(x_train_dataloader,y_train_dataloader,x_test_dataloader,y_test_dataloader,input_dim=input_dim, 
                                                        epoch=opt.epoch,
                                                        hidden_dim1=opt.hidden_dim1,
                                                        hidden_dim2=opt.hidden_dim2, 
                                                        hidden_dim3=opt.hidden_dim3,
                                                        hidden_dim4=opt.hidden_dim4,
                                                        num_layers=opt.num_layers,
                                                        output_dim=opt.output_dim,
                                                        num_stock='',
                                                        window=window,
                                                        lr=opt.lr,
                                                        batch_size=opt.batch_size,
                                                        dropout=opt.dropout,
                                                        shuffle=opt.shuffle,
                                                        patience=opt.patience,
                                                        delta=opt.delta,
                                                        batch_norm=opt.batch_norm,
                                                        mode=opt.style,
                                                        local_rank=opt.local_rank,
                                                        id=opt.id)
        # pdb.set_trace()
        # start=sub_range[0]
        # end=sub_range[-1]
        # y_train_date=pd.Series(flat_list(y_date_train)) # date*10,10=batch_size
        # y_test_date=pd.Series(flat_list(y_date_test))
        # pdb.set_trace()
        y_test_date=get_date_df(y_date_test)
        train_pred=pd.Series(flat_list(train_pred)) #len==1870
        test_pred=pd.Series(flat_list(test_pred))
        # train_y=pd.Series(flat_list(y_train_dataloader))
        test_y=pd.Series(flat_list(y_test_dataloader))
        # _,train_year_ic,train_month_ic,train_day_ic=factor_ic(y_train_date,train_pred,train_y)
        # print("===============================Training data===============================")
        # print("{}-{} year ic -----------------".format(start,end))
        # print(train_year_ic)
        # print("{}-{} month ic -----------------".format(start,end))
        # print(train_month_ic)
        # # print("{}-{} daily ic -----------------".format(start,end))
        # # print(train_day_ic)
        # # test:
        # pdb.set_trace()
        test_factor,test_day_ic,day_ic_base,day_ic_base2=\
            factor_ic(y_test_date,test_pred,test_y,baseline=True) #,baseline=True
        # pdb.set_trace()
        print("===============================Testing data===============================")
        # print("{}-{} year ic -----------------".format(start,end))
        # print(test_year_ic)
        # print("{}-{} month ic -----------------".format(start,end))
        # print(test_month_ic)
        # print("{}-{} daily ic -----------------".format(start,end))
        # print(test_day_ic)
        # pdb.set_trace()
        # df=pd.concat([month_ic_base,test_month_ic],axis=1,keys=['baseline','test']) 
        df=pd.concat([test_day_ic,day_ic_base.baseline1_daily_ic,day_ic_base2.baseline2_daily_ic],axis=1)
        res_df=pd.concat([res_df,df])
        factor_df=factor_df.append(test_factor)
        # pdb.set_trace()
    
    res_df.to_csv('./result/res_ic_{}_{}.csv'.format(factor,opt.year)) 
    factor_df.to_csv('./factors/_{}_{}_df.csv'.format(factor,opt.year))

def nn_score_training(ret):
    data=get_nn_score()
    # data.corr()
    data['pred']=data.sum(axis=1)
    data=pd.concat([data,ret],axis=1)
    return data

if __name__ == "__main__":
    # data=pv_factor().sort_values(by=['code','date'])
    # pdb.set_trace()
    # data=get_nn_score()
    # pdb.set_trace()
    # t1=time.time()
    # opt=parse_opt()
    # set_seed(opt.seed)
    # dynamic_training(opt,factor='minutely_bar')
    # dynamic_training(opt,factor='intra_return')
    # dynamic_training(opt,factor='pv_factor')
    # dynamic_training(opt,factor='money_flow_factor')
    # dynamic_training(opt,factor='weekly_bar')
    # dynamic_training(opt,factor='daily_bar')
    # t2=time.time()
    # s=t2-t1
    # m=s/60
    # h=m/60
    # print('程序运行时长共{}h'.format(h)) 
    opt=parse_opt() # 解析参数s
    set_seed(opt.seed) # 设置随机种子
    day_price=pd.read_csv("./data/day_price2016-2023.csv",names=['code','date','money','open','high','low','close','vol'])
    """
    day_price拉取的sql语句:
        select b.TICKER_SYMBOL,b.TRADE_DATE,a.TURNOVER_VALUE,OPEN_PRICE_1,HIGHEST_PRICE_1,LOWEST_PRICE_1,CLOSE_PRICE_1,b.TURNOVER_VOL
        from datayes.dbo.mkt_equd a join datayes.dbo.mkt_equd_adj b
        on a.TICKER_SYMBOL=b.TICKER_SYMBOL and a.TRADE_DATE=b.TRADE_DATE
        where b.TRADE_DATE between '2016-01-01' and '2023-12-31'
        order by b.TICKER_SYMBOL,b.TRADE_DATE
    """
    date_range,ret_df=get_test_date(day_price,year=opt.year,time_len=opt.time_len) #从21/01/04-23/12/06
    print('训练数据从{}到{}, 固定{}年一次独立训练,每隔{}天一训'.format(date_range[0],date_range[-1],opt.time_len,opt.gap))
    # 读取全时间段的沪深300成份股(对应拉到的data)
    stocks_df=sql_read(date_range,pool=opt.pool) #,pool='上证50','全A'
    # pdb.set_trace()
    # dynamic_training(opt,factor='minutely_bar',day_price=day_price,date_range=date_range,ret_df=ret_df,stocks_df=stocks_df)
    timing=[]
    for factor in ['minutely_bar']:  #,'intra_return','daily_bar','pv_factor','money_flow_factor','weekly_bar','minutely_bar'
        t1=time.time()
        # faulthandler.enable()
        print('=========================== {} ==========================='.format(factor))
        if opt.style=='dynamic':
            # pdb.set_trace()
            dynamic_training(opt,factor=factor,day_price=day_price,date_range=date_range,ret_df=ret_df,stocks_df=stocks_df)
        # elif opt.style=='static':
        #     static_training(opt,factor=factor)y
        
        t2=time.time()
        print('程序运行时长共{}h'.format((t2-t1)/360))
        timing.append((t2-t1)/360)
    # print(timing)
```



## config.py

```python
import argparse

def parse_opt():
    """
    parser=argparse.ArgumentParser() :创建一个解析对象
    parser.add_argument(): 向对象中添加要关注的命令行参数和选项
    parser.parse_args(): 调用parse_args()方法进行解析
    """
    parser=argparse.ArgumentParser()
    parser.add_argument('--style',type=str,default='dynamic')
    parser.add_argument('--start',type=int,default=2021)
    parser.add_argument('--end',type=int,default=2023)
    parser.add_argument('--year',type=str,default='2017')
    parser.add_argument('--id',type=int,default=0)
    
    # related to data processing 
    # parser.add_argument('--window',type=int,default=30)
    parser.add_argument('--output',type=int,default=1)
    parser.add_argument('--percent',type=float,default=0.9175)
    parser.add_argument('--gap',type=int,default=22)
    parser.add_argument('--time_len',type=float,default=1.0)

    # related to hyper paramter
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--epoch',type=int,default=100)
    parser.add_argument('--num_layers',type=int,default=10)
    parser.add_argument('--hidden_dim1',type=int,default=128)
    parser.add_argument('--hidden_dim2',type=int,default=64)
    parser.add_argument('--hidden_dim3',type=int,default=32)
    parser.add_argument('--hidden_dim4',type=int,default=16)
    parser.add_argument('--output_dim',type=int,default=1)
    parser.add_argument('--batch_size',type=int,default=64)
    parser.add_argument('--dropout',type=float,default=0)

    parser.add_argument('--shuffle',type=bool,default=False) # dataloader参数
    parser.add_argument('--patience',type=int,default=5) # earlystopping参数
    parser.add_argument('--delta',type=float,default=0) # earlystopping参数
    
    parser.add_argument('--batch_norm',type=bool,default=True)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--pool',type=str,default='全A')
    
    # ===============
    parser.add_argument('--local_rank', type=int,default=0) 

    args=parser.parse_args()
    return args
```



## *pytorchtools.py

附加文件

```python
import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

```



## neutralize.py

没有用到这个文件，用了中性化，模型效果会变差

```python
import pymssql
import pandas as pd
import time
import numpy as np
import datetime
from collections import OrderedDict
from processing import *
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# 中性化处理==============================================================================================================
def neutrlize_factors(date,start_date,end_date,code_list_str,server='10.60.0.165',
                                                      user='read_only',
                                                      password='read_only',
                                                      database='datayes'):
    # 拉到市值因子数据
    if len(code_list_str)==1:
        sql1="""select TRADE_DATE,NEG_MARKET_VALUE,TICKER_SYMBOL from datayes.dbo.mkt_equd_eval where TRADE_DATE
        between '{}' and '{}' and TICKER_SYMBOL = {} order by TRADE_DATE,TICKER_SYMBOL""".format(start_date,end_date,code_list_str[0])
        #--------------------------------------------------------------------------------------
        sql2="""select a.PARTY_ID,left(a.TYPE_ID,8),b.type_name,a.INTO_DATE,a.OUT_DATE 
        from datayes.[dbo].[md_inst_type] a join [datayes].[dbo].[md_type] b on left(a.TYPE_ID,8)=b.TYPE_ID 
        where a.TYPE_ID like '010321%' and (a.out_date IS NULL or a.out_date>='{}') 
        order by a.PARTY_ID,a.INTO_DATE""".format(start_date)
        #--------------------------------------------------------------------------------------
        sql3="""SELECT distinct(a.security_id), a.sec_short_name, a.ticker_symbol, a.type_name, a.into_date, a.out_date, b.party_id,a.EXCHANGE_CD,b.LIST_DATE 
        FROM [datayes].[dbo].[md_sec_type] a left join [datayes].[dbo].[md_security] b on a.security_id = b.security_id 
        where a.type_name= '全部A股' and a.ticker_symbol = {} order by a.ticker_symbol""".format(code_list_str[0])
    else:
        sql1="""select TRADE_DATE,NEG_MARKET_VALUE,TICKER_SYMBOL from datayes.dbo.mkt_equd_eval where TRADE_DATE
        between '{}' and '{}' and TICKER_SYMBOL in {} order by TRADE_DATE,TICKER_SYMBOL""".format(start_date,end_date,tuple(code_list_str))
        #--------------------------------------------------------------------------------------
        sql2="""select a.PARTY_ID,left(a.TYPE_ID,8),b.type_name,a.INTO_DATE,a.OUT_DATE 
        from datayes.[dbo].[md_inst_type] a join [datayes].[dbo].[md_type] b on left(a.TYPE_ID,8)=b.TYPE_ID 
        where a.TYPE_ID like '010321%' and (a.out_date IS NULL or a.out_date>='{}') 
        order by a.PARTY_ID,a.INTO_DATE""".format(start_date)
        #--------------------------------------------------------------------------------------
        sql3="""SELECT distinct(a.security_id), a.sec_short_name, a.ticker_symbol, a.type_name, a.into_date, a.out_date, b.party_id,a.EXCHANGE_CD,b.LIST_DATE 
        FROM [datayes].[dbo].[md_sec_type] a left join [datayes].[dbo].[md_security] b on a.security_id = b.security_id 
        where a.type_name= '全部A股' and a.ticker_symbol in {} order by a.ticker_symbol
        """.format(tuple(code_list_str))
    conn=pymssql.connect(server,user,password,database)
    if conn:
        print('Get the size factor...')
        size_factor=pd.read_sql(sql1,conn)
        print('Get the sector factor...')
        stock_indu=pd.read_sql(sql2,conn).rename(columns={'PARTY_ID':'party_id', '':'type_id','INTO_DATE':'into_date','OUT_DATE':'out_date'})
        stocklist=pd.read_sql(sql3,conn,columns = ['security_id', 'sec_short_name', 'ticker_symbol', 'type_name', 'into_date', 'out_date', 'party_id','exchange','ipo_date'])
        conn.commit()
        conn.close()  
        #数据简单处理-----------------------------
        print('Processing the size data...')
        size_factor=size_factor.rename(columns={'TRADE_DATE':'date','NEG_MARKET_VALUE':'size','TICKER_SYMBOL':'code'})
        size_factor.code=size_factor.code.astype('int') 
        size_factor.date=size_factor.date.astype('str')
        #---------------------------------------------
        print('Processing the sector data...')
        t1=time.time()
        stocklist_df = OrderedDict()
        for idate in sorted(list(set(date['date'].iloc[1:]))):
            idate=datetime.date(*map(int, idate.split('-')))
            temp=stocklist[ (stocklist['into_date'].notnull()) & (stocklist['into_date'] <= idate)\
                            & ((stocklist['out_date'].isnull()) | (stocklist['out_date']>idate))][['security_id','party_id','ticker_symbol','LIST_DATE']].reset_index(drop=True)
            temp['indu_name']=stock_indu[(stock_indu.into_date<= idate)&((stock_indu.out_date>idate)|(stock_indu.out_date.isnull()))].\
    drop_duplicates(subset='party_id',keep='first', inplace=False).set_index('party_id').reindex(temp['party_id'])['type_name'].tolist()
            temp=temp.set_index('ticker_symbol')
            stocklist_df[idate]=temp    
        t2=time.time()
        print('[timing: {}s]'.format(t2-t1))   
        return size_factor,stocklist_df
    return "数据库连接失败"  

def standardize(ret,mode=1):
    # 1.去极值： IQR=Q3-Q1
    Q1=np.percentile(ret,25)
    Q3=np.percentile(ret,75)
    IQR=Q3-Q1
    upper=Q3+1.5*IQR
    lower=Q1-1.5*IQR
    ret=ret.clip(lower,upper)
    # 2.标准化
    if mode==1: #max,min
        min=ret.min()
        max=ret.max()
        ret=(ret-min)/(max-min)
    elif mode==2: # z-socre
        ret=z_score(ret)
    elif mode==3:
        ret=ret/10**np.ceil(np.log10(ret.abs().max()))
    else:
        return "重新输入参数"
    return ret


def neutral(sub_df,sector_factor,mode=1):
    # 去极值 & 标准化
    sub_df['size']=standardize(sub_df['size']) #对市值因子做标准化
    # sector处理
    # 根据code做合并
    sector=sector_factor.reset_index()
    sector.rename(columns={'ticker_symbol':'code'},inplace=True)
    sector['code']=sector.code.astype('int')
    dataset=sub_df.merge(sector,on='code',how='inner')[['ret','size','indu_name']]  
    # data.droplevel(level=0,axis=1) 
    # 变dummy[]
    sector_dummies=pd.get_dummies(dataset['indu_name'])
    dataset=pd.concat([dataset,sector_dummies],axis=1).drop(['indu_name'],axis=1)
    # 3.线性回归取残差
    X=dataset.drop(['ret'],axis=1)
    y=dataset['ret']
    result=sm.OLS(y,X.astype(float)).fit()
    res=z_score(result.resid) 
    return res


def mkt_neutral(data,size_factor,stocklist_df):
    print("neutralizing...")
    t1=time.time()
    tmp=pd.merge(data,size_factor,on=['date','code'])
    # 截面上做中性化,即同一天内对所有股票做中性化
    cols=['date','code','ret','size']
    df=tmp[cols]
    df_res=data[data.columns.drop(['ret']).to_list()]
    groups=df.groupby('date')
    for group,indices in groups.groups.items():
        sector=stocklist_df[datetime.date(*map(int,group.split('-')))] #根据分组（日期），选择当天对应的sector factor
        sub_df=df.loc[indices,['ret','size','code']] 
        after_neutral=neutral(sub_df,sector)
        df.loc[indices,'ret']=after_neutral.values
    data=pd.merge(df,df_res)
    t2=time.time()
    print('[timing: {}s]'.format(t2-t1))
    return data
```





```python

"""
# 股票代码求交集==============================================================================================================
# def get_set_code(data):
    # 写错的，没删，留个思路
    # df=data[['date','code']]
    # while not (df.groupby('date')['code'].nunique() == 1).all(): #判断所有日期下的code是否相同数量
        # groups=df.groupby('date')
    # min_num_code=df.groupby('date')['code'].count().min() 
        # for g in groups:
        #     if len(g[1])==min_num_code:
        #         code_list=set(g[1].iloc[:,1].to_list())
        #         break
        # df=pd.DataFrame(groups.apply(lambda x:x[x.iloc[:,1].astype('str').isin(code_list)]).values).rename(columns={0:'date',1:'code'})
        # 对code做交集
    # grouped=df.groupby('date')['code'].apply(lambda x:x.iloc[:min_num_code]).reset_index()[['date','code']] 
    # len(set(grouped.reset_index().groupby('date')['code'].count().values)) 
    # df=pd.merge(data,grouped,on=['date','code'])   
    # pass
"""
```



# 处理方式

日度（资金流 / pv_factor / 周度）

> （1）从16-24年滚动 1年/1.5年/2年 一次独立训练，然后每次独立训练期间，每隔22天（gap）训练一次
>
> （2）根据 close 数据，计算5日，10日，20日移动平均，高开低手，成交量，成交额，用复权因子乘以价格，成交量/复权因子，得到真实价格数据；预测值用open计算
>
> （3）数据切分（自写dataloader函数）：从每天开始向上取30个交易日，每只股票，取的样本点；根据这30天最后一天对日期进行辨别，也就是识别出日期是否超过date_range的[-gap了]，不是就是train，是那么test（为了每一天都能预测到）
>
> （4）神经网络里的数据：10个特征，全A（5600多个票），相当于一个batch就是 5600 x 30 x 10，同一batch内做归一化（不能有未来数据）
>
> （5）模型搭建：4层lstm，一层简易的attention (自写)，128-> 64 ->32 -> 16，训练结束将test_pred拉出来
>
> （6）用日期，股票代码，z_score(test_pred) -> factor， 真实值 => 做corr计算，得到ic值 
>
> 整个程序用过两台服务器跑，4张1080，单机多卡运行；2张4090，以防cpu爆炸，分年训，不同年份训练在不同gpu卡上。参数写在外面，命令行调用



分钟（intra_return）

> （1）从16-24年滚动 1年/1.5年/2年 一次独立训练，然后每次独立训练期间，每隔22天（gap）训练一次
>
> （2）特征生成如下所示（intra_return）
>
> （3）数据切分（自写dataloader函数）：从每天开始向上取5个交易日，每只股票，取的样本点；drop掉了9:30的数据
>
> （4）神经网络里的数据：特征数，全A（5600多个票），batch -> 5600 x ( 5x8 ) x 特征数，同一batch内做归一化

```python
    df=pd.read_hdf('./data/30M_price.h5')
    df=df.sort_values(by=['code','date','time'])
    df['intraday_ret']=df.groupby('code').apply(lambda x: x.close/x.close.shift(1)-1).values
    early_ret=df.groupby(['code','date']).apply(lambda x: np.nan if len(x)<2 else x.intraday_ret.iloc[1]) #早盘30分钟收益率
    data=early_ret.reset_index().rename(columns={0:'early_ret'})
    data['open_ret']=df.groupby(['code','date']).apply(lambda x:x.intraday_ret.iloc[0]).values #开盘收益率
    data['max_ret']=df.groupby(['code','date']).apply(lambda x: np.nan if len(x)==1 else max(x.intraday_ret.iloc[1:])).values
    data['min_ret']=df.groupby(['code','date']).apply(lambda x: np.nan if len(x)==1 else min(x.intraday_ret.iloc[1:])).values
    data['avg_ret']=df.groupby(['code','date']).apply(lambda x: np.nan if len(x)==1 else np.mean(x.intraday_ret.iloc[1:])).values
    data['last_ret']=df.groupby(['code','date']).apply(lambda x : np.nan if len(x)!=9 else x.intraday_ret.iloc[-1]).values
```



模型搭建

> loss函数用了mse和自写的corr
>
> lr 大小对收敛频率影响
>
> 用了earlystopping和dropout，后面dropout设为0了

遇到问题：

> 1. 刚开始处理数据时，不清楚金融数据是一只股票相当于是一句话，后来自己画图+代码，理解了
> 2. 遇到nan的问题（代码里已经处理过nan数据了），后面debug从loss开始，发现出现了inf，再退回数据本身发现y中存在inf
> 3. 配环境的时候，遇到过明明有gpu，但装的包cuda is availiable返回false，看了看自己装的包，发现一个中间包是cpu用，手动装成cuda版本，再装包发现询问将包变成cpu版，停止安装，换了个版本
>
> 教训：下次写这种代码最好分批写。。。



