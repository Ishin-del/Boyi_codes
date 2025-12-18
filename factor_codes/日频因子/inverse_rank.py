
"""
反向量价排序乘积因子: https://zhuanlan.zhihu.com/p/1984561152991188280
"""
import feather
import pandas as pd

class Inverse_rank:
    def __init__(self,start=None,end=None,daily_path=None):
        self.start =start or '20220101'
        self.end =end or '20251201'
        self.daily_path=daily_path or r'Z:\local_data\Data_Storage\daily.feather'
        self.daily_df=feather.read_dataframe(self.daily_path)
        self.daily_df=self.daily_df[(self.daily_df['DATE']>=self.start)&(self.daily_df['DATE']<=self.end)].reset_index(drop=True)

    def cal(self):
        self.daily_df=self.daily_df[['DATE','TICKER','close','volume']]
        self.daily_df.sort_values(['DATE','TICKER'],inplace=True)
        # -----------------------------
        self.daily_df['close_roll10_std']=self.daily_df.groupby('TICKER')['close'].transform(lambda x:x.rolling(10,5).std())
        self.daily_df['close_roll10_std_rank']=self.daily_df.groupby('DATE')['close_roll10_std'].rank(pct=True,method='first')
        self.daily_df['closeRollStdRankAdjust']=1-self.daily_df['close_roll10_std_rank']
        self.daily_df['volume_rank']=self.daily_df.groupby('DATE')['volume'].rank(pct=True,method='first')
        self.daily_df['inverseRank']=-1*self.daily_df['close_rank']*self.daily_df['volume_rank']
        # self.daily_df['volStable']=self.daily_df['volume_rank']*self.daily_df['closeRollStdRankAdjust']
        return self.daily_df[['TICKER','DATE','inverseRank']]
        # return self.daily_df[['TICKER','DATE','volStable']]

    def run(self):
        df=self.cal()
        feather.write_dataframe(df,r'C:\Users\admin\Desktop\inverseRank.feather')
        # feather.write_dataframe(df,r'C:\Users\admin\Desktop\volStable.feather')

if __name__=='__main__':
    obj=Inverse_rank()
    obj.run()