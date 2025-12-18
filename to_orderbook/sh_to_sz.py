# @Author: Yixin Tian
# @File: sh_to_sz.py
# @Date: 2025/10/13 18:29
# @Software: PyCharm
import pandas as pd
def cal(x,l):
    x=x.reset_index(drop=True)
    if x[l+'OrderNo'].iloc[0]==0:
        return
    position=x[x.Type=='A'].index
    if position.empty:
        qty=x.Qty.sum()
        if l=='Buy':
            p = x.Price.max()
        else:
            p=x.Price.min()
    else:
        qty=x.loc[:position[-1]].Qty.sum()
        p = x[x.Type=='A'].Price.iloc[0]
    t=x.TickTime.iloc[0]
    return pd.Series([qty,p,t])
# 20250729
df=pd.read_feather(r"\\192.168.1.28\i\data_feather(2025)\data_feather\20250729\stock_tick\600111.feather")
df=df[(df.Type=='T')|(df.Type=='A')]
df.drop(columns=['index','ChannelNo'],inplace=True)
# df[df['BuyorderNo']== 181436]
buy=df.groupby(['BuyOrderNo'],as_index=False).apply(lambda x: cal(x,l='Buy')).rename(columns={'BuyOrderNo':'order',0:'qty',1:'price',2:'time'})
buy['side']='buy'
sell=df.groupby(['SellOrderNo'],as_index=False).apply(lambda x: cal(x,l='Sell')).rename(columns={'SellOrderNo':'order',0:'qty',1:'price',2:'time'})
sell['side']='sell'
res=pd.concat([buy,sell]).dropna(axis=0).reset_index(drop=True) #.to_csv(r"C:\Users\Administrator\Desktop\tt.csv",index=False)
print(res)