import os
import pandas as pd
from joblib import Parallel,delayed
from tqdm import tqdm


def close_vwap(df):
    if 'Type' in df.columns: # 24年以后的上海数据
        df.rename(columns={'Qty':'TradeQty','TickTime':'tradetime','TickIndex':'ApplSeqNum'},inplace=True)
        df['ExecType'] = 'F'
        df['tradedate'] = df.tradetime.astype(str).str[:8]
        flag='*'
    elif 'TradeIndex' in df.columns:# 24年以前的上海数据
        df.rename(columns={'TradePrice':'Price','TradeTime':'tradetime','TradeIndex':'ApplSeqNum'},inplace=True)
        df['ExecType']='F'
        df['tradedate']=df.tradetime.astype(str).str[:8]
    df=df[['Price','TradeQty','tradetime','ExecType','ApplSeqNum','SecurityID','tradedate']]
    df = df[df.ExecType == 'F']
    df['time']=df.tradetime.astype(str).str[8:12]
    df=df[(df.time>='1455')&(df.time<'1457')]
    try:
        close_vwap=sum(df.Price*df.TradeQty)/sum(df.TradeQty)
    except ZeroDivisionError:
        return
    return close_vwap

def main(date,code):
    if code == '603176':
        return
    path = f'h\data_feather({date[:4]})' if str(date)[:4] < '2024' else f'i\data_feather({date[:4]})'
    try:
        if code[-2:]=='SH':
            if 'tick' in os.listdir(fr'\\192.168.1.28\{path}\data_feather\{date}'):
                df = pd.read_feather(fr'\\192.168.1.28\{path}\data_feather\{date}\tick\{code[:6]}.feather')
            else:
                df = pd.read_feather(fr'\\192.168.1.28\{path}\data_feather\{date}\stock_tick\{code[:6]}.feather')
                df=df[df.Type == 'T']
        else:
            df = pd.read_feather(fr'\\192.168.1.28\{path}\data_feather\{date}\hq_trade_spot\{code[:6]}.feather')
    except FileNotFoundError:
        return
    res=close_vwap(df)
    print(date,code)
    return [code,date,res]

if __name__=='__main__':
    stock_pool = pd.read_feather(r'D:\tyx\数据\min_total_mv_400_stock(1).feather')
    stock_pool = stock_pool[stock_pool.DATE.str[:4] != '2021']
    # res={'TICKER':[],'DATE':[],'close_vwap':[]}
    # for _, r in tqdm(stock_pool.iterrows()):
    #     tmp_res=main(r.DATE,r.TICKER)
    #     if tmp_res is None:
    #         continue
    #     res['TICKER'].append(r.TICKER)
    #     res['DATE'].append(r.DATE)
    #     res['close_vwap'].append(tmp_res)
    tmp=Parallel(n_jobs=24)(delayed(main)(r.DATE,r.TICKER) for _,r in stock_pool.iterrows())
    df = pd.DataFrame([x for x in tmp if x is not None])
    df.columns = ['code', 'date', 'close_vwap']
    df.to_csv(r'C:\Users\Administrator\Desktop\close_vwap.csv')