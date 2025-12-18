import os
import feather
import numpy as np
import pandas as pd
import warnings
from joblib import Parallel, delayed
from tqdm import tqdm

#对于每只股票的子df,这里的df_sub是经过TICKER和DATE排序并索引重置后的结果，计算日频GTR
def cal_GTR(df_sub):
    df_sub['GTR']=df_sub['turnover_rate'].rolling(window=20).std()
    #print(df_sub)
    return df_sub[['DATE','TICKER','GTR']]

class GTR:
    def __init__(self, start, end, calender_path=None, df_path=None, savepath=None, float_mv_path=None):
        self.start = start or '20220104'
        self.end = end or '20251121'
        self.df_path = df_path or r'D:\zkh\Price_and_vols_at_auction_time\concat_daily'
        self.calender_path = calender_path or r'\\192.168.1.210\Data_Storage2'
        self.savepath = savepath or rf'D:\因子保存\{self.__class__.__name__}'
        self.tmp_path = os.path.join(self.savepath, 'basic')
        self.float_mv_path=float_mv_path or r'Z:\local_data\Data_Storage'


        os.makedirs(self.savepath, exist_ok=True)
        os.makedirs(str(self.tmp_path), exist_ok=True)
        # self.daily_list_file = os.listdir(r'D:\zkh\Price_and_vols_at_auction_time\concat_daily')
        # self.split_sec = split_sec
        calendar = pd.read_csv(os.path.join(self.calender_path, 'calendar.csv'), dtype={'trade_date': str})
        calendar.rename(columns={'trade_date': 'DATE'}, inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']

    def __daily_calculation(self):
        if os.path.exists(os.path.join(self.tmp_path,'auc_free_turnover.feather')):
            df_old = feather.read_dataframe(os.path.join(self.tmp_path,'auc_free_turnover.feather'))
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
            # exist_date=
        else:
            df_old = pd.DataFrame()
            exist_date = []
        max_min_date = max(os.listdir(self.df_path)).replace('.feather', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)
        # print(date_list)
        def get_daily(date):
            warnings.filterwarnings('ignore')
            filename = date + '.feather'
            if not os.path.exists(os.path.join(self.df_path, filename)):
                if date <= max_min_date:
                    raise FileNotFoundError
            else:
                dt = feather.read_dataframe(os.path.join(self.df_path, f'{date}.feather'))
                dt = dt[(dt['tradetime'] >= 92000000) & (dt['tradetime'] < 92459950)]
                dt = dt.sort_values(by=['TICKER', 'ApplSeqNum']).reset_index(drop=True)
                split_by_sec = dt.groupby(['TICKER']).agg(
                    {'当前撮合成交量': ['first', 'last'], 'Price': ['first', 'last', 'max', 'min']}).reset_index()
                split_by_sec.columns = ['TICKER', 'vol_first', 'vol_last', 'open', 'close', 'high', 'low']
                split_by_sec['volume_diff'] = split_by_sec['vol_last'] - split_by_sec['vol_first']
                split_by_sec['DATE'] = date
                split_by_sec.drop(columns=['vol_first', 'vol_last'], inplace=True)
                dt = split_by_sec.reset_index(drop=True)
                dt.rename(columns={'volume_diff': 'volume'}, inplace=True)
                return dt

        df = Parallel(n_jobs=10)(delayed(get_daily)(date) for date in tqdm(date_list))
        df = pd.concat(df).reset_index(drop=True)
        df=df[['TICKER','DATE','volume']]

        df_float_mv=feather.read_dataframe(os.path.join(self.float_mv_path,'float_mv.feather'))
        df_float_mv.sort_values(['TICKER','DATE'],inplace=True)
        df_float_mv['float_mv']=df_float_mv.groupby('TICKER')['float_mv'].shift(1) #流通市值用前一天进行
        df = pd.merge(df, df_float_mv, on=['TICKER', 'DATE'], how='inner')
        df['free_turnover']=df['volume']/df['float_mv']
        df=df[['TICKER','DATE','free_turnover']]
        alls = pd.concat([df_old, df]).reset_index(drop=True)
        feather.write_dataframe(alls,os.path.join(self.tmp_path,'auc_free_turnover.feather'))
        # print(alls)

    def cal(self):
        warnings.filterwarnings('ignore')
        df=feather.read_dataframe(os.path.join(self.tmp_path,'auc_free_turnover.feather'))
        turnover_rate = df.groupby('TICKER').apply(lambda x:x['free_turnover'] / x['free_turnover'].shift(1) - 1)
        '''
        因为代码里有这个x['free_turnover'].shift(1)，所以代码参数里的第一天肯定是没数据的，也没法用市场中位数填充
        '''
        df['turnover_rate'] = turnover_rate.reset_index(level = 'TICKER', drop = True)
        df['turnover_rate'] = df['turnover_rate'].replace([np.inf, -np.inf], np.nan)
        df=df[['TICKER','DATE','turnover_rate']]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df['turnover_rate'] = df['turnover_rate'].fillna(df.groupby('DATE')['turnover_rate'].transform('median'))
        df.rename(columns={'turnover_rate':'auc_turnover_rate'},inplace=True)
        feather.write_dataframe(df, os.path.join(self.savepath,'auc_turnover_rate.feather'))
        # print(df)

    def run(self):
        self.__daily_calculation()
        self.cal()
    ''' 
    GTR = df.groupby('TICKER').apply(lambda x:x['turnover_rate'].rolling(window=20,min_periods=5).std())
    按天roll.std
    '''
if __name__ == '__main__':
    obj=GTR(start='20220104',end='20220115',
        df_path=r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time\concat_daily',
            savepath=r'C:\Users\admin\Desktop')
    obj.run()