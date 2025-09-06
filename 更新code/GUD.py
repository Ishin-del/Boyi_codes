import os
import signal
import time
import feather
import numpy as np
import pandas as pd
import datetime as dt
import warnings
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
import statsmodels.formula.api as smf
from statsmodels.api import OLS
# from test_tools.cyy_funcs import kill_mutil_process
from tool_tyx.path_data import DataPath
"""  开源证券：日内分钟收益率的时序特征：逻辑讨论与因子增强——市场微观结构研究系列（19）
      时序特征：跌幅时间重心偏离因子"""

# 计算涨幅时间重心的函数
def cal_GUD(df_sub):
    UR = (df_sub['time_num'] * df_sub['ret']).sum()
    R_norm = df_sub['ret'].abs().sum()
    G = UR / R_norm
    ret_mean = df_sub['ret'].mean()
    return pd.Series([G, ret_mean], index = ['G','ret_mean'])

# 截面回归：以跌幅时间重心作为被解释变量，对涨幅时间重心回归取残差DropTimeGravityDeviation
# 输入对DATE进行分组后的df
def cal_DTGD(df_sub):
    # model = smf.ols(df_sub['G_d'], data=df_sub).fit()
    df_sub = df_sub.dropna(subset = ['G_d','G_u'])
    if df_sub.empty:
        return
    model = OLS(df_sub['G_d'], df_sub['G_u']).fit()
    # df_sub['DTGD'] = model.resid
    return pd.Series(model.resid, index= df_sub.index)


def cal_TGD(df_sub):
    # 第一步：进行第一次回归，得到残差项𝜀𝑢和𝜀𝑑
    df_sub = df_sub.dropna(subset = ['G_d','G_u','R1','R2','ret_on','ret_mean_u','ret_mean_d'])
    if df_sub.empty:
        return
    df_sub['const'] = 1
    df_out = df_sub[['const']].copy()
    try:
        df_out['Epsilon_u'] = OLS(df_sub['G_u'], df_sub[['R1','R2','ret_on','ret_mean_u','const']]).fit().resid
        df_out['Epsilon_d'] = OLS(df_sub['G_d'], df_sub[['R1','R2','ret_on','ret_mean_d','const']]).fit().resid
        df_out['TGD'] = OLS(df_out['Epsilon_d'], df_out[['const','Epsilon_u']]).fit().resid
    except:
        print()
    return df_out['TGD']


class GUD:
    def __init__(self, start=None, end=None, data_path=None, min_data_path=None, savepath=None, testpath=None):
        self.start_date = start #or '20250102'
        self.end_date = end #or '20250127'
        # todo:
        self.data_path = DataPath.to_path
        # self.min_data_path = min_data_path or '\\\\10.36.35.85\\Data_storage\\Min_Data'
        self.sh_min_path = DataPath.sh_min
        self.sz_min_path = DataPath.sz_min
        exist_file_list = os.listdir(self.sh_min_path)
        self.max_min_data_date = max([x[:10] for x in exist_file_list])
        self.tmp_path=DataPath.tmp_path
        self.savepath = savepath
        # os.makedirs(self.savepath, exist_ok=True)
        # # os.makedirs(os.path.join(self.savepath, 'factor'), exist_ok=True)
        # os.makedirs(os.path.join(self.savepath, 'basic'), exist_ok=True)
        # os.makedirs(self.testpath, exist_ok=True)

    def Generate_Calendar(self):
        df_calendar = pd.read_csv(os.path.join(self.data_path, 'calendar.csv'), dtype={'trade_date': str}) # 日期数据
        df_calendar.rename(columns={'trade_date':'DATE'},inplace=True)
        df_calendar = df_calendar[(df_calendar['DATE'] >= self.start_date) & (df_calendar['DATE'] <= self.end_date)]
        df_calendar = df_calendar.sort_values(by='DATE', ascending=True)
        # 生成交易日期序列
        self.date_list = df_calendar['DATE']

    def load_min_data(self, file_name):
        warnings.filterwarnings('ignore')
        pid = os.getpid() #当前进程的PID
        # 加载一分钟数据并计算每天的因子
        try:
            df_min1 = feather.read_dataframe(os.path.join(self.sh_min_path, file_name))
            df_min2=feather.read_dataframe(os.path.join(self.sz_min_path, file_name))
        except FileNotFoundError:
            return
        df_min=pd.concat([df_min1,df_min2]).reset_index(drop=True)[['TICKER','DATE','open','close','min','volume']]
        # 剔除每天vol求和为0,收盘价为0的股票
        df_min['ret'] = df_min['close'] / df_min['open'] - 1
        # 对每只股票生成时间标识序列
        # df_min['time_num'] = df_min['trade_time'].factorize()[0]+1
        df_min['time_num'] = df_min['min'].factorize()[0]+1
        # df_sig = df_min.groupby('ts_code').agg({'open':'min','volume':'sum','time_num':len})
        df_min = df_min[(df_min['min'] > 1459) | (df_min['min'] < 1457)]
        df_sig = df_min.groupby('TICKER').agg({'open':'min','volume':'sum','time_num':len})
        # todo: 239
        valid_ts_codes = df_sig[(df_sig['open'] >0) & (df_sig['volume'] >0) & (df_sig['time_num'] ==239)]
        df_min = df_min[df_min['TICKER'].isin(valid_ts_codes.index)]

        df_min1 = df_min[df_min['ret'] > 0]
        df_min2 = df_min[df_min['ret'] < 0]

        df_min_10 = df_min[df_min['time_num'] <= 31]
        df_min_1030 = df_min[(df_min['time_num'] > 31) & (df_min['time_num'] <=61)]
        # tqdm.pandas()
        df_factor1 = df_min1.groupby(['TICKER']).apply(cal_GUD).rename(columns = {'G':'G_u','ret_mean':'ret_mean_u'})
        df_factor2 = df_min2.groupby(['TICKER']).apply(cal_GUD).rename(columns = {'G':'G_d','ret_mean':'ret_mean_d'})

        df_R1 = df_min_10.groupby(['TICKER'])['ret'].mean()
        df_R2 = df_min_1030.groupby(['TICKER'])['ret'].mean()

        df_out = pd.concat([df_factor1, df_factor2, df_R1.rename('R1'), df_R2.rename('R2')], axis = 1)
        if df_out.empty:
            return None,pid
        df_out['time_diff'] = df_out['G_u'] - df_out['G_d']
        # df_out['time_dist'] = df_out['time_diff'].abs()
        df_out['TICKER'] = df_out.index
        # df_out['DATE'] = df_min['trade_time'].iloc[0][:10].replace('-','')
        df_out['DATE'] = df_min['DATE'].iloc[0]
        # df_min.replace([np.inf, -np.inf], np.nan, inplace=True)
        # # print(df_min)
        # res_df = df_min[['TICKER', 'DATE', 'G_u', 'G_d', 'time_diff', 'time_dist']]
        # res_df.dropna(subset=['G_u', 'G_d'], inplace=True)  # 回归的两列不能有缺失值
        # res_df = res_df.drop_duplicates().reset_index(drop=True)
        return df_out, pid

    def cal_daily(self):
        try:
            factor_old = feather.read_dataframe(os.path.join(self.tmp_path,'basic.feather'))
            exist_date = factor_old['DATE'].unique()
        except:
            factor_old = pd.DataFrame()
            exist_date = []

        # file_list = [x[:4] + '-' + x[4:6] + '-' + x[6:8] + '.feather' for x in np.setdiff1d(self.date_list, exist_date)]
        file_list = [x+ '.feather' for x in np.setdiff1d(self.date_list, exist_date)]
        if len(file_list) == 0:
            print('无需更新basic')
        else:
            dummies = Parallel(n_jobs=10)(
                delayed(self.load_min_data)(file_name) for file_name in
                tqdm(file_list, desc='Calculating Daily G_u,G_d,time_diff')) #,time_dist

            factor_ls =[x[0] for x in dummies]
            # pid_list =[x[1] for x in dummies]
            # self.kill_mutil_process(set(pid_list))
            if factor_ls!=[None]:
                factor_df = pd.concat(factor_ls)
                factor_old = pd.concat([factor_old, factor_df])
                factor_old.reset_index(drop=True,inplace=True)
                feather.write_dataframe(factor_old, os.path.join(self.tmp_path,'basic.feather'))
        daily = feather.read_dataframe(os.path.join(DataPath.to_df_path,'daily.feather'),columns=['TICKER','DATE','close'])
        # adj_df=feather.read_dataframe(os.path.join(DataPath.to_df_path,'adj_factors.feather'))
        daily.sort_values(['TICKER','DATE'],inplace=True)
        daily['pre_close']=daily.groupby('TICKER')['close'].shift(1)
        daily = daily[(daily['DATE'] <= self.end_date) & (daily['DATE'] >= self.start_date)]
        daily['ret_on'] = daily['close'] / daily['pre_close'] - 1
        daily = daily[['TICKER','DATE','ret_on']]

        factor_old = pd.merge(daily, factor_old, on=['TICKER','DATE'], how='left')
        self.GUD_daily = factor_old.sort_values(by=['DATE','TICKER']).reset_index(drop=True)

    def cal_F(self):
        '''# self.GUD_daily应该是从头到尾 每次跑一遍'''
        self.GUD_daily = self.GUD_daily.sort_values(by = ['TICKER','DATE'])

        tqdm.pandas(desc='Calculating DTGD')
        DTGD = self.GUD_daily.set_index('TICKER').groupby('DATE').progress_apply(cal_DTGD)
        self.GUD_daily = pd.merge(self.GUD_daily, DTGD.rename('DTGD'), on=['TICKER','DATE'], how='left')

        tqdm.pandas(desc='Calculating TGD')
        TGD = self.GUD_daily.set_index('TICKER').groupby('DATE').progress_apply(cal_TGD)
        self.GUD_daily = pd.merge(self.GUD_daily, TGD.rename('TGD'), on=['TICKER','DATE'], how='left')

        tqdm.pandas(desc='Calculating rolling_mean of DTGD')
        self.GUD_daily['DTGD'] = self.GUD_daily.groupby('TICKER')['DTGD'].progress_apply(
            lambda x: x.rolling(window=20, min_periods=5).mean()).values  # 对20天窗口求均值

        tqdm.pandas(desc='Calculating rolling_mean of TGD')
        self.GUD_daily['TGD'] = self.GUD_daily.groupby('TICKER')['TGD'].progress_apply(
            lambda x: x.rolling(window=20, min_periods=5).mean()).values  # 对20天窗口求均值

        self.GUD_daily['G_u'] = self.GUD_daily.groupby('TICKER')['G_u'].apply(
            lambda x: x.rolling(window=20, min_periods=5).mean()).values
        self.GUD_daily['G_d'] = self.GUD_daily.groupby('TICKER')['G_d'].apply(
            lambda x: x.rolling(window=20, min_periods=5).mean()).values
        self.GUD_daily['time_diff'] = self.GUD_daily.groupby('TICKER')['time_diff'].apply(
            lambda x: x.rolling(window=20, min_periods=5).mean()).values
        # self.GUD_daily['time_dist'] = self.GUD_daily.groupby('TICKER')['time_dist'].apply(
        #     lambda x: x.rolling(window=20, min_periods=15).mean()).values

        out_put_factors = ['G_u','G_d','time_diff', 'DTGD','TGD'] #,'time_dist'
        for col in out_put_factors:
            df_out = self.GUD_daily[['TICKER', 'DATE',col]]
            print(df_out)
            feather.write_dataframe(df_out, os.path.join(DataPath.save_path_update, 'TGD_'+col + '.feather'))
            feather.write_dataframe(df_out, os.path.join(self.savepath, 'TGD_' + col + '.feather'))
        # df1=self.GUD_daily[['TICKER', 'DATE','G_u']]
        # df2=self.GUD_daily[['TICKER', 'DATE','G_d']]
        # df3=self.GUD_daily[['TICKER', 'DATE','time_diff']]
        # df4=self.GUD_daily[['TICKER', 'DATE','time_dist']]
        # df5=self.GUD_daily[['TICKER', 'DATE','DTGD']]
        # df6=self.GUD_daily[['TICKER', 'DATE','TGD']]
        # return df1,df2,df3,df4,df5,df6

    def run(self):
        self.Generate_Calendar()
        self.cal_daily()
        self.cal_F()
        # df1,df2,df3,df4,df5,df6=self.cal_F()
        # return df1,df2,df3,df4,df5,df6
def update(today='20250822'):
    print('GUD系列因子更新中')
    GUD(start='20200101', end=today, savepath=DataPath.factor_out_path).run()

if __name__ == '__main__':
    update()