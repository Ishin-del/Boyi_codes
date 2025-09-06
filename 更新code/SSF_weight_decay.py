import os
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib.parallel import Parallel, delayed, cpu_count
import warnings

from tool_tyx.path_data import DataPath


# 月耀眼收益率 SSF 版本2：半衰期合成
# 对应研报 ： 20220412-方正证券-多因子选股系列研究之一：成交量激增时刻蕴含的alpha信息


class SSF_half(object):
    def __init__(self, start='20250102', end='20250228', path=None, savepath=None, halflife=10,test_path=None,data = None):
        self.data = data
        self.testpath = test_path or 'O:\\VP_speed_up\\'
        self.halflife = halflife
        self.start = start
        self.end = end
        self.tmp_path = DataPath.tmp_path
        self.path = DataPath.to_path
        self.savepath = savepath
        self.daily_list_file = os.listdir(DataPath.sh_min)
        calendar = pd.read_csv(self.path + '\\calendar.csv', dtype={'trade_date': str})
        calendar.rename(columns={'trade_date': 'DATE'}, inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']

    def __daily_calcuation(self):
        if os.path.exists(self.tmp_path + '\\basic_data2.feather'):
            df_old = feather.read_dataframe(self.tmp_path + '\\basic_data2.feather')
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []

        # max_min_date = max(self.daily_list_file)[:10].replace('-', '')
        max_min_date = max(self.daily_list_file).replace('.feather', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)

        # **************    主要函数部分,获取日内的数据
        def get_date(x):
            # 1 求分钟交易额的差值
            x = x.sort_values(by=['min'])
            xdif = np.diff(x['volume'])
            if xdif.size == 0:
                return
            threshold = xdif.mean() + xdif.std()
            x = x.iloc[1:, :]
            x['mvdif'] = xdif
            x = x.reset_index(drop=True)
            g = x[x['mvdif'] > threshold].index.tolist()

            # 先将不存在日耀波动率的股票的值设0
            # 月耀眼波动率如果在这一天不存在的话,就证明当天的股价波动小
            # 最后要求均值的话如果为全零填充也将会拉低月均耀眼波动率,这与实际相符
            a = 0
            if g:
                a = x['ret'].iloc[g].mean()
            # return [1,2,3]
            return [x['TICKER'].unique()[0],x['DATE'].unique()[0], a]

        def get_daily(date):  # for循环tqdm
            warnings.filterwarnings('ignore')
            # filename = '-'.join([date[:4], date[4:6], date[6:8]]) + '.feather'
            filename = date + '.feather'
            if not os.path.exists(os.path.join(DataPath.sh_min,filename)):
                if date <= max_min_date:
                    raise FileNotFoundError
            else:
                to_day = np.array(['TICKER', 'DATE', 'ssf'])
                dt1 = feather.read_dataframe(os.path.join(DataPath.sh_min, filename))  # 去除开盘和收盘的时间
                dt2 = feather.read_dataframe(os.path.join(DataPath.sz_min, filename))  # 去除开盘和收盘的时间
                dt = pd.concat([dt1, dt2]).reset_index(drop=True)[['TICKER', 'DATE', 'min', 'volume', 'open', 'close']]
                # dt['min'] = [i.split(' ')[1] for i in dt['trade_time'].tolist()]
                dt = dt.drop(index=dt[dt['min'] == 930].index)
                dt = dt.drop(index=dt[dt['min'] == 1458].index)
                dt = dt.drop(index=dt[dt['min'] ==1459].index)
                dt = dt.drop(index=dt[dt['min'] == 1500].index)

                dt = dt.drop(index=dt[np.isclose(dt['open'], 0)].index)
                dt['ret'] = dt['close'] / dt['open'] - 1
                dt = dt.dropna()

                dt = dt.groupby('TICKER').apply(get_date)
                dd = []
                for i in dt:
                    if i:
                        dd.append(i)
                to_day = np.vstack((to_day, dd))
                del dt, dd
                to_day = pd.DataFrame(to_day[1:], columns=to_day[0]) if to_day.shape != (3,) else pd.DataFrame()
                to_day['DATE'] = [''.join(i.split('-')) for i in to_day['DATE'].tolist()]
                to_day['ssf'] = to_day['ssf'].astype(float)
                return to_day
            # *****************

        # newdata = []
        # for i in tqdm(date_list, desc='calculating daily'):
        #     newdata.append(get_daily(i))
        newdata = Parallel(n_jobs=10)(delayed(get_daily)(i) for i in tqdm(date_list,desc='calculating daily'))
        if len(newdata) > 0:
            to_day = pd.concat(newdata)
            if not df_old.empty:
                df_old['ssf'] = df_old['ssf'].astype(float)
            df_old = pd.concat([df_old, to_day]).sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
            df_old.to_feather(self.tmp_path + '\\basic_data2.feather')

    def cal_SSF(self):

        def dataframe_ewm_rolling(dataframe: pd.DataFrame, halflife: int = 1, min_periods: int = 1):
            if dataframe.shape[0] < min_periods:
                col = dataframe.columns
                return pd.DataFrame({i: np.nan for i in col}, index=[dataframe.index[-1]])
            warnings.filterwarnings('ignore')
            numeric_df = dataframe._get_numeric_data()
            # cols = numeric_df.columns
            # idx = dataframe.index[-1]
            # mat = numeric_df.to_numpy(dtype=float, na_value=np.nan, copy=False)
            if min_periods is None:
                min_periods = 1
            # correl = hyx.ewm(mat, halflife=halflife, min_periods=min_periods)
            # return pd.DataFrame(correl, columns=[idx], index=cols).T
            correl = numeric_df.ewm(halflife=halflife, min_periods=min_periods).mean().iloc[-1, :].T
            if not isinstance(correl, pd.DataFrame):
                correl = correl.to_frame().T
            return correl


        def ewm_(dataframe: pd.DataFrame, halflife: int = 1, min_periods: int = 1, window: int = 1, verbose=True):
            # 输入一个透视表，输出一个透视表
            from tqdm import trange
            if dataframe.empty:
                raise ValueError('DataFrame is empty')
            res = []
            if verbose:
                for i in trange(1, dataframe.shape[0] + 1, desc='Processing'):
                    now = dataframe.iloc[max(i - window, 0):i, :]
                    res.append(dataframe_ewm_rolling(now, halflife=halflife, min_periods=min_periods))
            else:
                for i in range(1, dataframe.shape[0] + 1):
                    now = dataframe.iloc[max(i - window, 0):i, :]
                    res.append(dataframe_ewm_rolling(now, halflife=halflife, min_periods=min_periods))
            res = pd.concat(res)
            return res


        data = feather.read_dataframe(self.tmp_path + '\\basic_data2.feather')
        pt = data.pivot_table(values='ssf', columns='TICKER', index='DATE')
        pt = pt.ffill()
        pp = pt.melt(ignore_index=False)
        pp = pp.reset_index(drop=False)
        pp = pp.dropna().reset_index(drop=True)
        pp = pp[['TICKER', 'DATE']]
        data = pd.merge(pp, data, how='left', on=['TICKER', 'DATE'])
        data = data.reset_index(drop=True)
        data = data[(data['DATE'] <= self.end) & (data['DATE'] >= self.start)]
        half = self.halflife

        # data.to_feather(self.savepath + f'factor\\MSF_half{half}test.feather')
        if os.path.exists(self.savepath + f'\\SSF_half{half}.feather'):
            dz = feather.read_dataframe(self.savepath + f'\\SSF_half{half}.feather')
            dz = dz[dz['DATE'] <= self.end]
            dtls = dz['DATE'].unique()
            dtls.sort()
            data = data[data['DATE'] > min(self.end,dtls[max(len(dtls) - 66,0)])]
        data = data.reset_index()

        def getmean(x):
            x['abs'] = abs(x['ssf'] - x['ssf'].mean())
            return x

        tqdm.pandas(desc='step1 on calculating SSF')
        data = data.groupby('DATE').progress_apply(getmean)
        data = data[['TICKER', 'DATE', 'abs']].reset_index(drop=True)
        #
        # def get_SSF(x):
        #     x['mean_Vol_daily'] = x['abs'].rolling(20, min_periods=10).apply(
        #         lambda t: t.ewm(halflife=half).mean().iloc[-1])
        #     x['std_Vol_daily'] = x['abs'].rolling(20, min_periods=10).apply(
        #         lambda t: t.ewm(halflife=half).std().iloc[-1])
        #     x[f'SSF_{half}'] = (x['mean_Vol_daily'] + x['std_Vol_daily']) / 2
        #     return x
        #
        # # tqdm.pandas(desc='step2 on calculating MSF')
        # # data = data.groupby('TICKER').progress_apply(get_MSF)
        # newdata = Parallel(n_jobs=2)(delayed(get_SSF)(data[data['TICKER'] == i])
        #                                        for i in tqdm(data['TICKER'].unique(), desc='step2 on calculating SSF'))
        # data = newdata[0]
        # for i in tqdm(range(1, len(newdata)), desc='concating DataFrame'):
        #     data = pd.concat([data, newdata[i]])
        # data = data.reset_index(drop=True)

        pt1 = data.pivot_table(columns=['TICKER'], index=['DATE'], values='abs', dropna=False)
        pp1 = ewm_(pt1, halflife=half, min_periods=10, window=20)
        pz1 = pp1.melt(ignore_index=False).reset_index(drop=False)
        pz1.columns = ['DATE', 'TICKER', 'mean_Vol_daily']

        def get_MSF(x):
            x['std_Vol_daily'] = x['abs'].rolling(20, min_periods=10).apply(
                lambda t: t.ewm(halflife=half).std().iloc[-1])
            return x

        newdata = Parallel(n_jobs=10)(delayed(get_MSF)(data[data['TICKER'] == i]) for i in tqdm(data['TICKER'].unique(), desc='step2 on calculating SSF'))
        # data = newdata[0]
        # data = data.append(newdata[1:])
        data=pd.concat(newdata)
        data = pd.merge(data, pz1, how='left', on=['TICKER', 'DATE'])
        data[f'SSF_{half}'] = (data['mean_Vol_daily'] + data['std_Vol_daily']) / 2
        data = data.reset_index(drop=True)
        data = data[['TICKER', 'DATE', f'SSF_{half}']].reset_index(drop=True)

        # data = data[['TICKER','DATE',f'SSF_{half}']]
        if os.path.exists(self.savepath + f'\\SSF_half{self.halflife}.feather'):
            dtz = feather.read_dataframe(self.savepath + f'\\SSF_half{self.halflife}.feather')
            # dtz = dtz[dtz['DATE'] <= self.end].reset_index(drop=True)
            # dtdate = dtz['DATE'].unique()
            # dtdate.sort()
            data = data.reset_index(drop=True)
            # data_ = data[data['DATE'] >= dtdate[-20]]
            # dtzz = dtz[dtz['DATE'] < dtdate[-20]]
            data_ = data[~data['DATE'].isin(dtz.DATE.unique())]
            data_ = pd.concat([dtz, data_])
            print(data_)
            feather.write_dataframe(data_, DataPath.save_path_update + f'\\SSF_half{self.halflife}.feather')
            feather.write_dataframe(data_, self.savepath + f'\\SSF_half{self.halflife}.feather')
        else:
            datanow = data
            print(datanow)
            feather.write_dataframe(datanow, DataPath.save_path_old + f'\\SSF_half{self.halflife}.feather')
            feather.write_dataframe(datanow, DataPath.save_path_update + f'\\SSF_half{self.halflife}.feather')
        print(f'SSF_half{self.halflife} has been saved')
        print(np.setdiff1d(data.columns, ['TICKER', 'DATE']), sorted(data['DATE'].unique())[-1])

    def run(self):
        """
        方便外部调用
        在统一的全部因子更新脚本中，对于每个import的类都会只运行run这个方法，以实现因子更新的目的
        """
        # if not os.path.exists(self.savepath):
        #     os.makedirs(self.savepath)
        #     os.makedirs(self.savepath + '程序\\')
        #     os.makedirs(self.savepath + 'factor\\')
        #     os.makedirs(self.savepath + 'basic\\')
        self.__daily_calcuation()
        self.cal_SSF()
def update(today='20250820'):
    object = SSF_half(start='20250701', end=today, savepath=DataPath.factor_out_path)
    object.run()

if __name__ == '__main__':
    update()