import os
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib.parallel import Parallel, delayed, cpu_count
import shutil
from acc import hyx
import warnings


# 月耀眼波动率 MSF 版本2 半衰期
# 对应研报 ： 20220412-方正证券-多因子选股系列研究之一：成交量激增时刻蕴含的alpha信息
# 计算过程

# 定义“激增时刻”的这一分钟及其随后的 4 分钟，是因成交 量激增而引起投资者关注的 5 分钟。投资者对成交量激增的反应
# 在这 5 分钟里表现得最充分最强烈，我们将这 5 分钟称为“耀眼 5 分钟”。

# 使用分钟收盘价，计算每分钟的收益率，进而可以得到每个“耀眼 5 分钟”里，收益率的标准差，
# 作为成交量激增引起的价格波动率，我们将其称为“耀眼波动率”。

# 1.  我们使用分钟收盘价，计算每分钟的收益率，进而可以得到每个“耀眼 5 分钟”里，收益率的标准差，
# 作为成交量激增引起的价格波动率，我们将其称为“耀眼波动率”。
# 2  我们计算 A 股票在 t 日内所有“耀眼波动率”的均值，作为 t 日 A股票对成交量的激增在波动层面上反应的代理变量，
# 记为“日耀眼波动率”。
# 3  根据前述分析，我们希望“日耀眼波动率”不要太大，也不要太小，适度最好，为了不引入其他参数，
# 我们此处选取“日耀眼波动率”的截面均值作为最“适度”的水平。因此我们将每日的“日耀眼波动率”
# 减去截面的均值再取绝对值，表示个股的“日耀眼波动率”与市场平均水平的距离，并将其记为日频因子“适度日耀眼波动率”。
# 4  我们分别计算最近 20 个交易日的“适度日耀眼波动率”的平均值和标准差，
# 记为“月均耀眼波动率”因子和“月稳耀眼波动率”因子。
# 5  将“月均耀眼波动率”与“月稳耀眼波动率”等权合成，得到“月耀眼波动率”因子。

class MSF_half(object):
    def __init__(self, start='20130104', end='20240820', path=None, savepath=None, halflife=10, test_path=None):
        self.testpath = test_path or '\\\\10.36.35.106\\Factor_Storage\\周楷贺\\原始数据\\拟使用量价因子\\'
        self.halflife = halflife
        self.start = start
        self.end = end
        self.path = path or '\\\\10.36.35.85\\Data_Storage\\Factor_Storage\\'
        self.savepath = savepath or 'E:\\vp_zkh\\vp_factors\\MSF\\'
        self.daily_list_file = os.listdir('\\\\10.36.35.85\\Data_Storage\\Min_Data\\')
        calendar = pd.read_csv(self.path + 'Calendar.csv', dtype={'DATE': str})
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']

    def __daily_calcuation(self):
        if os.path.exists(self.savepath + 'basic\\basic_data.feather'):
            df_old = feather.read_dataframe(self.savepath + 'basic\\basic_data.feather')
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []

        max_min_date = max(self.daily_list_file)[:10].replace('-', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)

        def get_date(x):
            xdif, retdiff = np.diff(x['vol']), np.diff(x['ret'])
            if xdif.size == 0:
                return
            threshold = xdif.mean() + xdif.std(),
            x = x.iloc[1:, :]
            x['retdiff'], x['mvdif'] = retdiff, xdif
            x = x.reset_index(drop=True)
            g, res = x[x['mvdif'] > threshold].index, []
            for i in g:
                if i < x.shape[0] - 5:
                    res.append(x['retdiff'].iloc[i:i + 5].std())
            # 先将不存在日耀波动率的股票的值设0
            # 月耀眼波动率如果在这一天不存在的话,就证明当天的股价波动小
            # 最后要求均值的话如果为全零填充也将会拉低月均耀眼波动率,这与实际相符
            a = sum(res) / len(res) if res else 0
            return [x['ts_code'].unique()[0],
                    np.array(x['trade_time'].iloc[0:1])[0][:10], a]

        def get_daily(date):
            # for循环tqdm
            filename = '-'.join([date[:4], date[4:6], date[6:8]]) + '.feather'
            if not os.path.exists('\\\\10.36.35.85\\Data_Storage\\Min_Data\\' + filename):
                if date <= max_min_date:
                    raise FileNotFoundError
            else:
                to_day = np.array(['TICKER', 'DATE', 'msf'])
                dt = feather.read_dataframe('\\\\10.36.35.85\\Data_Storage\\Min_Data\\' + filename)  # 去除开盘和收盘的时间
                dt['min'] = [i.split(' ')[1] for i in dt['trade_time'].tolist()]
                dt = dt.drop(index=dt[dt['min'] == '09:30:00'].index)
                dt = dt.drop(index=dt[dt['min'] == '14:58:00'].index)
                dt = dt.drop(index=dt[dt['min'] == '14:59:00'].index)
                dt = dt.drop(index=dt[dt['min'] == '15:00:00'].index)
                # 去除不开盘的股票
                dt = dt.drop(index=dt[np.isclose(dt['open'], 0)].index)
                dt = dt.dropna()
                dt['ret'] = dt['close'] / dt['open'] - 1
                dt = dt.sort_values(by=['ts_code', 'trade_time'])
                dt['diff'] = dt.groupby('ts_code')['vol'].diff()
                dt = dt.dropna()

                dt = dt.groupby('ts_code').apply(get_date)
                dd = []
                for i in dt:
                    if i:
                        dd.append(i)
                to_day = np.vstack((to_day, dd))
                del dt, dd
                to_day = pd.DataFrame(to_day[1:], columns=to_day[0]) if to_day.shape != (3,) else pd.DataFrame()
                to_day['DATE'] = [''.join(i.split('-')) for i in to_day['DATE'].tolist()]
                to_day['msf'] = to_day['msf'].astype(float)
                return to_day
            # *****************

        newdata = []
        for i in tqdm(date_list):
            newdata.append(get_daily(i))  # for循环 tqdm
        if newdata:
            to_day = pd.concat(newdata).reset_index(drop=True)
            if not df_old.empty:
                df_old['msf'] = df_old['msf'].astype(float)
            df_old = pd.concat([df_old, to_day]).sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
            df_old.to_feather(self.savepath + 'basic\\basic_data.feather')

    def cal_MSF_half(self):

        def dataframe_ewm_rolling(dataframe: pd.DataFrame, halflife: int = 1, min_periods: int = 1):
            if dataframe.shape[0] < min_periods:
                col = dataframe.columns
                return pd.DataFrame({i: np.nan for i in col}, index=[dataframe.index[-1]])
            warnings.filterwarnings('ignore')
            numeric_df = dataframe._get_numeric_data()
            cols = numeric_df.columns
            idx = dataframe.index[-1]
            mat = numeric_df.to_numpy(dtype=float, na_value=np.nan, copy=False)
            if min_periods is None:
                min_periods = 1
            correl = hyx.ewm(mat, halflife=halflife, min_periods=min_periods)
            return pd.DataFrame(correl, columns=[idx], index=cols).T

        def ewm_mean(dataframe: pd.DataFrame, halflife: int = 1, min_periods: int = 1, window: int = 1, verbose=True):
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

        data = feather.read_dataframe(self.savepath + 'basic\\basic_data.feather')
        pt = data.pivot_table(values='msf', columns='TICKER', index='DATE')
        pt = pt.ffill()
        pp = pt.melt(ignore_index=False)
        pp = pp.reset_index(drop=False)
        pp = pp.dropna().reset_index(drop=True)
        pp = pp[['TICKER', 'DATE']]
        data = pd.merge(pp, data, how='left', on=['TICKER', 'DATE'])
        data = data[(data['DATE'] <= self.end) & (data['DATE'] >= self.start)]
        data = data.reset_index(drop=True)

        half = self.halflife
        if os.path.exists(self.savepath + f'factor\\MSF_half{half}.feather'):
            dz = pd.read_feather(self.savepath + f'factor\\MSF_half{half}.feather')
            dz = dz[dz['DATE'] <= self.end]
            dtls = dz['DATE'].unique()
            dtls.sort()
            data = data[data['DATE'] > min(self.end, dtls[max(len(dtls) - 66, 0)])]
        data = data.reset_index()

        def getmean(x):
            x['abs'] = abs(x['msf'] - x['msf'].mean())
            return x

        tqdm.pandas(desc='step1 on calculating MSF')
        data = data.groupby('DATE').progress_apply(getmean)
        data = data[['TICKER', 'DATE', 'abs']].reset_index(drop=True)

        pt1 = data.pivot_table(columns=['TICKER'], index=['DATE'], values='abs', dropna=False)
        pp1 = ewm_mean(pt1, halflife=half, min_periods=10, window=20)
        pz1 = pp1.melt(ignore_index=False).reset_index(drop=False)
        pz1.columns = ['DATE', 'TICKER', 'mean_Vol_daily']
        data = data[['TICKER', 'DATE', 'abs']]

        ptable = data.pivot_table(columns='TICKER', index='DATE', values='abs')

        def ewmstd(x, half, outname):
            beta = [1]
            if not isinstance(half, int):
                half = x.shape[0] // 2
            subalpha = 0.5 ** (1 / (half))

            for i in range(1,x.shape[0]):
                beta.append(beta[i-1]*subalpha)
            beta.reverse()
            beta = np.array(beta)

            x = (x.T * beta).T
            cnt = x.count().reset_index(drop=False)
            cnt = cnt[cnt[0] >= half]['TICKER'].tolist()
            x = x[cnt]
            xstd = x.std()
            xstd = xstd.reset_index(drop=False)
            xstd = xstd.rename(columns={0: outname})
            xstd['DATE'] = x.index.max()
            return xstd[['TICKER', 'DATE', outname]]

        ewm_std = []
        for idx in tqdm(range(20, ptable.shape[0] + 1)):
            now = ptable.iloc[idx - 20:idx, :]
            ewm_std.append(ewmstd(x=now, half=half, outname='std_Vol_daily'))

        ewm_std = pd.concat(ewm_std).reset_index(drop=True)

        # ((p * beta[-3:]).std() / (beta[-1] + beta[-2] + beta[-3])).std()
        # p.ewm(halflife=half).std()
        # ll = pd.Series([1,2,3,4])
        # x1 = ll[:2].std()
        # x2 = ll[:3].std()
        # y2 = ll.ewm(halflife=10).std().iloc[-2]
        # bb = (x2-y2)/(y2-x1)
        #
        # def get_MSF(x):
        #     x['std_Vol_daily'] = x['abs'].rolling(20, min_periods=10).apply(
        #         lambda j: j.ewm(halflife=half).std().iloc[-1])
        #     return x

        # newdata = Parallel(n_jobs=2)(
        #     delayed(get_MSF)(data[data['TICKER'] == i])
        #     for i in tqdm(data['TICKER'].unique(), desc='step2 on calculating MSF'))

        # tqdm.pandas(desc='step2 on calculating MSF')
        # newdata = data.groupby('TICKER').progress_apply(get_MSF).reset_index(drop=True)

        # data = pd.concat(newdata).reset_index(drop=True)
        data = pd.merge(ewm_std, pz1, how='left', on=['TICKER', 'DATE'])
        data[f'MSF_{half}'] = (data['mean_Vol_daily'] + data['std_Vol_daily']) / 2
        data = data.reset_index(drop=True)
        data = data[['TICKER', 'DATE', f'MSF_{half}']].reset_index(drop=True)
        # 将最终日期设置为22年7月底，然后把下面一行打开，然后比较重合部分(上一次减90加回溯天数，
        # 大概三个月-上一次最后一天)是否对的上
        # data.to_feather(self.savepath + f'factor\\MSF_half{half}test.feather')

        if os.path.exists(self.savepath + f'factor\\MSF_half{half}.feather'):
            dtz = pd.read_feather(self.savepath + f'factor\\MSF_half{half}.feather')
            dtz = dtz[dtz['DATE'] <= self.end].reset_index(drop=True)
            dtdate = dtz['DATE'].unique()
            dtdate.sort()
            data = data.reset_index(drop=True)
            data_ = data[data['DATE'] >= dtdate[-20]]
            dtzz = dtz[dtz['DATE'] < dtdate[-20]]
            data_ = pd.concat([dtzz, data_])
            feather.write_dataframe(data_, self.savepath + f'factor\\MSF_half{half}.feather')
            feather.write_dataframe(data_,
                                    self.testpath + f'MSF_half{half}.feather')
        else:
            feather.write_dataframe(data, self.savepath + f'factor\\MSF_half{half}.feather')
            feather.write_dataframe(data,
                                    self.testpath + f'MSF_half{half}.feather')
        print(f'MSF_half{half} has been saved')
        print(np.setdiff1d(data.columns, ['TICKER', 'DATE']), sorted(data['DATE'].unique())[-1])

    def run(self):
        """
        方便外部调用
        在统一的全部因子更新脚本中，对于每个import的类都会只运行run这个方法，以实现因子更新的目的
        """
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
            os.makedirs(self.savepath + '程序\\')
            os.makedirs(self.savepath + 'factor\\')
            os.makedirs(self.savepath + 'basic\\')
        self.__daily_calcuation()
        self.cal_MSF_half()


if __name__ == '__main__':
    MSF_object = MSF_half()
    # break_point = 1
    MSF_object.run()
