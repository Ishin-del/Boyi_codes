import datetime as dt
import os
import time
import warnings

import feather
import numpy as np
import pandas as pd
import psutil
import statsmodels.api as sm
from joblib import Parallel, delayed
# from tqdm_joblib import tqdm_joblib
from tqdm import tqdm

from tool_tyx.path_data import DataPath
from tool_tyx.tyx_funcs import kill_mutil_process

warnings.filterwarnings("ignore")


def z_score(x):
    return (x-x.mean())/x.std()


class 花隐林间_Improved:
    def __init__(self, start=None, end=None,savepath=None):
        self.start_date = start
        self.end_date = end
        print(self.end_date)
        # WIN
        self.data_path = DataPath.to_path
        self.tmp_path=DataPath.tmp_path
        self.sz_min =DataPath.sz_min
        self.sh_min =DataPath.sh_min
        self.savepath=savepath
        exist_file_list = os.listdir(self.sz_min)
        self.max_min_data_date = max([x[:10] for x in exist_file_list])
        self.look_back_period = 20

        # 中间变量储存
        self.regression_new = None
        self.regression_df = None
        self.t_intercept_df = None
        self.factor_1_daily = None
        self.factor_2_daily = None
        self.factor_3_old = None

    def Generate_Calendar(self):
        df_calendar = pd.read_csv(self.data_path + '//calendar.csv',dtype={'trade_date': str})
        df_calendar.rename(columns={'trade_date':'DATE'},inplace=True)
        df_calendar = df_calendar[(df_calendar['DATE'] >= self.start_date)& (df_calendar['DATE'] <= self.end_date)]
        df_calendar['month'] = df_calendar['DATE'].map(lambda x: x[:6])
        df_calendar = df_calendar.sort_values(by=['month', 'DATE'],ascending=True)
        df_month_end = df_calendar.drop_duplicates(subset=['month'],keep='last')['DATE']
        self.date_list = df_calendar['DATE']
        self.month_end_list = df_month_end

    # def OLS(self, min_close, min_vol) -> np.ndarray:
    #     """return: [t-intercept, t5, t4, t3, t2, t1, t0, F]"""
    #     N = min_close.shape[0]
    #     # min_return = min_close[1:] / min_close[:-1] - 1
    #     min_return=min_close.pct_change()
    #     # min_vol_incre = min_vol[1:] - min_vol[:-1]
    #     min_vol_incre=min_vol.diff()
    #     indexer = np.arange(6)[None, :] + np.arange(N - 6)[:, None]
    #     X = sm.add_constant(min_vol_incre[indexer])
    #     Y = min_return[5:]
    #     try:
    #         model = sm.OLS(Y, X).fit()
    #         return np.append(model.tvalues, [model.fvalue])
    #     except RuntimeWarning:
    #         null_vector = np.empty(shape=(8,))
    #         null_vector[:] = np.nan
    #         return null_vector
    def OLS(self, min_close, min_vol) -> np.ndarray:
        """更清晰的实现"""
        returns = pd.Series(min_close.pct_change()).dropna()
        # returns*=100
        # vol_changes = min_vol.diff().dropna()
        # vol_changes = pd.Series(min_vol.pct_change()).dropna()
        vol_changes = pd.Series(min_vol.diff()).dropna()
        min_len = min(len(returns), len(vol_changes))
        returns = returns[-min_len:]
        vol_changes = vol_changes[-min_len:]
        if min_len < 7:  # 需要至少7个数据点
            return np.full(8, np.nan)
        # 创建滞后特征
        X_data = pd.DataFrame({
            'vol_t0': vol_changes,
            'vol_t1': vol_changes.shift(1),
            'vol_t2': vol_changes.shift(2),
            'vol_t3': vol_changes.shift(3),
            'vol_t4': vol_changes.shift(4),
            'vol_t5': vol_changes.shift(5)
        })
        # todo: 标准化：
        for c in X_data.columns.to_list():
            X_data[c]=z_score(X_data[c])
        aligned_data = pd.concat([returns, X_data], axis=1).dropna()
        aligned_data = aligned_data.replace([np.inf, -np.inf], np.nan)
        aligned_data.dropna(inplace=True)
        Y = aligned_data.iloc[:, 0]
        X = aligned_data.iloc[:, 1:]
        X = sm.add_constant(X)

        try:
            model = sm.OLS(Y, X).fit()
            return np.append(model.tvalues, [model.fvalue])
        except:
            return np.full(8, np.nan)

    def load_min_data(self, file_name):
        warnings.filterwarnings("ignore")
        pid = os.getpid()
        df_min_sz,df_min_sh=pd.DataFrame(),pd.DataFrame()
        # file_name=file_name.replace('-','')
        try:
            df_min_sz = feather.read_dataframe(os.path.join(self.sz_min,file_name),columns=['TICKER', 'DATE','close','volume','min'])
        except FileNotFoundError:
            # if file_name[:10] <= self.max_min_data_date:
            #     raise FileNotFoundError
            # else:
            pass
        try:
            df_min_sh = feather.read_dataframe(os.path.join(self.sh_min,file_name),columns=['TICKER', 'DATE','close','volume','min'])
        except FileNotFoundError:
            # if file_name[:10] <= self.max_min_data_date:
            #     raise FileNotFoundError
            # else:
            pass
        df_min=pd.concat([df_min_sz,df_min_sh]).reset_index(drop=True)
        if df_min.empty:
            return
        # print(df_min)
        df_min = df_min[df_min.groupby("TICKER")["volume"].transform(sum) != 0]
        df_min =df_min[~df_min['min'].isin([1457,1458,1459])]
        # df_min = df_min[df_min.groupby("TICKER")["close"].transform("count") == 238]
        df_min = df_min[df_min.groupby("TICKER")["close"].transform("count") >= 235]
        # print(df_min)
        if df_min.empty:
            return
        # date = df_min['trade_time'].iloc[0][:10]
        date=df_min['DATE'].iloc[0]
        # F_and_t_values = df_min.groupby("TICKER").apply(lambda x: self.OLS(x.close.to_numpy(), x.volume.to_numpy()))
        df_min.sort_values(['TICKER','min'],inplace=True)
        # df_min['min_ret']=df_min.groupby('TICKER')['close'].pct_change()
        # df_min['vol_rate']=df_min.groupby('TICKER')['volume'].pct_change()
        F_and_t_values = df_min.groupby("TICKER").apply(lambda x: self.OLS(x.close, x.volume))
        # F_and_t_values_df = pd.DataFrame(F_and_t_values.tolist(),columns=['t-intercept', 't5', 't4', 't3','t2', 't1',
        #                                  't0', "F-statistic"],index=F_and_t_values.index).reset_index()
        # print(F_and_t_values)
        res_df=pd.DataFrame(F_and_t_values.tolist(),columns=['t-intercept', 't5', 't4', 't3','t2', 't1',
                                         't0', "F-statistic"],index=F_and_t_values.index).reset_index()
        # F_and_t_values_df["DATE"] = date.replace("-", "")
        # F_and_t_values_df["DATE"] = date
        res_df['DATE']=date
        # print(res_df)
        return pid, res_df

    def 朝没晨雾(self):
        # todo:? 不去掉一个？
        factor = np.std(self.regression_df[["t5", "t4", "t3", "t2", "t1"]],axis=1)
        output = pd.DataFrame(factor.values,index=self.regression_df.index,columns=["朝没晨雾"])
        output["DATE"] = self.regression_df["DATE"].values
        output = output.reset_index().rename(columns={"index": "TICKER"})
        return output[["TICKER", "DATE", "朝没晨雾"]]

    def 午蔽古木(self):
        df = self.regression_df
        df = pd.merge(df,df.groupby("DATE", as_index=False)["F-statistic"].mean(),on="DATE",how="left",suffixes=("", "_mean"))
        factor = ((df["F-statistic"] > df["F-statistic_mean"]) * 2 -1) * np.abs(df["t-intercept"])
        output = pd.DataFrame(factor.values,index=self.regression_df.index,columns=["午蔽古木"])
        output["DATE"] = df["DATE"].values
        output = output.reset_index().rename(columns={"index": "TICKER"})
        return output[["TICKER", "DATE", "午蔽古木"]]

    def cal_corr20_daily(self, df):
        pid = os.getpid()
        df_20 = df.corr(min_periods=10)
        np.fill_diagonal(df_20.values, np.nan)
        df_20 = df_20.abs()
        corr = df_20.mean(axis=1)
        corr_df = pd.DataFrame(corr, columns=["夜眠霜路"])
        corr_df = corr_df.reset_index().rename(columns={"index": "TICKER"})
        corr_df["DATE"] = df.index[-1]
        return pid, corr_df[["TICKER", "DATE", "夜眠霜路"]]

    def 夜眠霜路(self):
        factor_list = Parallel(n_jobs=12)(delayed(self.cal_corr20_daily)(window)
            for window in tqdm(self.t_intercept_df.rolling(20), desc="cal 夜眠霜路",total=len(self.t_intercept_df) - 20 + 1))
        factor_list=[x for x in factor_list if x is not None]
        pid_list = [element[0] for element in factor_list]
        kill_mutil_process(list(set(pid_list)))

        factor_list = [element[1] for element in factor_list]
        factor_df = pd.concat(factor_list)
        return factor_df

    def cal_factor_daily(self):
        if os.path.exists(os.path.join(self.tmp_path,'花隐林间_回归结果.feather')):
            # todo：
            regression_df_old = feather.read_dataframe(os.path.join(self.tmp_path, '花隐林间_回归结果.feather'))
            cut_date = regression_df_old['DATE'].drop_duplicates().sort_values().iloc[-20]
            regression_df_old = regression_df_old[regression_df_old['DATE'] <= cut_date]
            exist_date = regression_df_old['DATE'].unique()

        else:
            # factor_3_old = pd.DataFrame(columns=["TICKER", "DATE", "夜眠霜路"])
            regression_df_old = pd.DataFrame(columns=['TICKER','t-intercept', 't5', 't4', 't3', 't2', 't1', 't0',"F-statistic"])
            exist_date = []

        # self.factor_3_old = factor_3_old
        updated_date = np.setdiff1d(self.date_list, exist_date)
        file_list=[x+ '.feather' for x in updated_date]
        # file_list = [x[:4] + '-' + x[4:6] + '-' + x[6:8] + '.feather' for x in updated_date]
        # print(file_list)
        if len(file_list) == 0:
            regression_df = pd.DataFrame()
        else:
            factor_daily_list = Parallel(n_jobs=12)(delayed(self.load_min_data)(file) for file in tqdm(file_list,desc="cal ols"))
            # print(factor_daily_list)
            factor_daily_list=[x for x in factor_daily_list if x is not None]
            pid_list = [element[0] for element in factor_daily_list]
            kill_mutil_process(list(set(pid_list)))
            factor_daily_list = [element[1] for element in factor_daily_list]
            print(factor_daily_list)
            regression_df = pd.concat(factor_daily_list)
            self.regression_new = regression_df.set_index("TICKER")

        regression_df = pd.concat([regression_df_old, regression_df])
        self.regression_df = regression_df.set_index("TICKER")

        # 重要中间变量: 回归结果
        print(regression_df)
        feather.write_dataframe(regression_df,os.path.join(self.tmp_path,'花隐林间_回归结果.feather'))
        factor_1_df = self.朝没晨雾()
        factor_2_df = self.午蔽古木()
        self.factor_1_daily = factor_1_df
        self.factor_2_daily = factor_2_df

        feather.write_dataframe(factor_1_df,self.savepath + '//朝没晨雾_daily.feather')
        feather.write_dataframe(factor_2_df,self.savepath + '//午蔽古木_daily.feather')

    def cal_factor_improved(self):
        factor_1_df, factor_2_df = self.factor_1_daily, self.factor_2_daily
        # factor_1_df["朝没晨雾"] = factor_1_df.groupby("TICKER", group_keys=False)["朝没晨雾"].apply(lambda x: x.rolling(10, min_periods=5).mean())
        # factor_2_df["午蔽古木"] = factor_2_df.groupby( "TICKER", group_keys=False)["午蔽古木"].apply(lambda x: x.rolling(10, min_periods=5).mean())
        # print(factor_1_df)
        # print(factor_2_df)
        # feather.write_dataframe(factor_1_df, self.savepath + '//朝没晨雾.feather')
        # feather.write_dataframe(factor_2_df, self.savepath + '//午蔽古木.feather')
        self.regression_df=feather.read_dataframe(os.path.join(self.tmp_path,'花隐林间_回归结果.feather')).set_index("TICKER")
        self.factor_2_daily=feather.read_dataframe(os.path.join(self.savepath,'午蔽古木_daily.feather'))
        print(self.regression_df)
        if os.path.exists(os.path.join(self.savepath +'//夜眠霜路.feather')):
            factor_3_old = feather.read_dataframe(os.path.join(self.savepath +'//夜眠霜路.feather'))
            cut_date = factor_3_old['DATE'].drop_duplicates().sort_values().iloc[-20]
            factor_3_old = factor_3_old[factor_3_old['DATE'] <= cut_date]
            begin_date = factor_3_old['DATE'].drop_duplicates().sort_values().iloc[-30]
        else:
            factor_3_old = pd.DataFrame(columns=['TICKER','DATE', '夜眠霜路'])
            begin_date = self.start_date

        regression_df_new = self.regression_df[self.regression_df['DATE']>=begin_date]
        t_intercept_df = regression_df_new[["t-intercept", "DATE"]]
        t_intercept_df = t_intercept_df.reset_index()
        t_intercept_df = pd.pivot_table(t_intercept_df,index="DATE",columns="TICKER")
        t_intercept_df.columns = t_intercept_df.columns.map(lambda x: x[1])
        t_intercept_df.index.name = None
        self.t_intercept_df = t_intercept_df
        print(self.t_intercept_df)
        factor_3_new = self.夜眠霜路()
        # print(factor_3_old)
        factor_3_new = factor_3_new[~factor_3_new["DATE"].isin(factor_3_old["DATE"].unique())]
        factor_3_df = pd.concat([factor_3_old, factor_3_new])

        factor_3_df = self.factor_2_daily.merge(factor_3_df, on=["DATE","TICKER"], how="left")[["TICKER", "DATE", "夜眠霜路"]]
        factor_3_df = factor_3_df.sort_values(by=["DATE", "TICKER"])
        print(factor_3_df)
        print(factor_3_df.dropna())
        feather.write_dataframe(factor_3_df, self.savepath + '//夜眠霜路.feather')

    def run(self):
        self.Generate_Calendar()
        # self.cal_factor_daily()
        self.cal_factor_improved()


if __name__ == '__main__':
    today = dt.datetime.today().strftime('%Y%m%d')
    now = time.time()

    terminal = 花隐林间_Improved(start='20200101', end='20250820',savepath=DataPath.save_path_old)
    # terminal = 花隐林间_Improved(start='20240927', end='20240927',savepath=DataPath.save_path_old)
    terminal.run()

    print('计算共用时%f秒' % (time.time() - now))
