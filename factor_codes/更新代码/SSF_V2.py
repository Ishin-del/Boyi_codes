import os
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
from tool_tyx.path_data import DataPath

class MSF_half(object):
    def __init__(self, start, end, halflife=10,savepath=None):
        self.halflife = halflife
        self.start = start
        self.end = end
        self.path = DataPath.to_path
        self.savepath = savepath
        self.tmp_path=DataPath.tmp_path
        self.daily_list_file = os.listdir(DataPath.sh_min)
        calendar = pd.read_csv(os.path.join(self.path,'calendar.csv'), dtype={'trade_date': str})
        calendar.rename(columns={'trade_date':'DATE'},inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']

    def __daily_calcuation(self):
        if os.path.exists(self.tmp_path + '\\ssf_v2中间数据.feather'):
            df_old = feather.read_dataframe(self.tmp_path + '\\ssf_v2中间数据.feather')
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []
        max_min_date = max(self.daily_list_file).replace('.feather', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)

        def get_date(x):
            xdif, retdiff = np.diff(x['volume']), np.diff(x['ret'])
            if xdif.size == 0:
                return
            # threshold1 = xdif.mean() + xdif.std()*3
            threshold2=xdif.mean()-xdif.std()*3
            x = x.iloc[1:, :] # 不要932的数据，因为xdif retdiff值为nan
            x['retdiff'], x['mvdif'] = retdiff, xdif
            x = x.reset_index(drop=True)
            # g= x[x['mvdif'] > threshold1].index.to_list()
            g= x[x['mvdif'] < threshold2].index.to_list()
            # g=x[(x['mvdif'] > threshold1)|(x['mvdif'] < threshold2)].index.to_list()
            a=x['ret'].iloc[g].mean() if g else 0
            return [x['TICKER'].unique()[0],x['DATE'].unique()[0], a] #np.array(x['trade_time'].iloc[0:1])[0][:10]

        def get_daily(date):
            warnings.filterwarnings('ignore')
            filename = date + '.feather'
            if not os.path.exists(os.path.join(DataPath.sh_min,filename)):
                if date <= max_min_date:
                    raise FileNotFoundError
            else:
                to_day = np.array(['TICKER', 'DATE', 'msf'])
                dt1 = feather.read_dataframe(os.path.join(DataPath.sh_min,filename))  # 去除开盘和收盘的时间
                dt2 = feather.read_dataframe(os.path.join(DataPath.sz_min,filename))  # 去除开盘和收盘的时间
                dt=pd.concat([dt1,dt2]).reset_index(drop=True)[['TICKER','DATE','min','volume','open','close']]
                dt = dt.drop(index=dt[dt['min'] == 930].index)
                dt = dt.drop(index=dt[dt['min'] == 1458].index)
                dt = dt.drop(index=dt[dt['min'] ==1459].index)
                dt = dt.drop(index=dt[dt['min'] == 1500].index)
                dt = dt.drop(index=dt[np.isclose(dt['open'], 0)].index)
                dt = dt.dropna()
                dt['ret'] = dt['close'] / dt['open'] - 1
                dt = dt.sort_values(by=['TICKER', 'min'])
                dt['diff'] = dt.groupby('TICKER')['volume'].diff()
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
                to_day['msf'] = to_day['msf'].astype(float)
                return to_day
        newdata=Parallel(n_jobs=14)(delayed(get_daily)(i) for i in tqdm(date_list))
        if newdata:
            to_day = pd.concat(newdata).reset_index(drop=True)
            if not df_old.empty:
                df_old['msf'] = df_old['msf'].astype(float)
            df_old = pd.concat([df_old, to_day]).sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
            df_old.to_feather(self.tmp_path + '\\ssf_v2中间数据.feather')
            # feather.write_dataframe(df_old,r'C:\Users\admin\Desktop\tmp.feather')

    def cal_MSF_half(self):
        def dataframe_ewm_rolling(dataframe: pd.DataFrame, halflife: int = 1, min_periods: int = 1):
            if dataframe.shape[0] < min_periods:
                col = dataframe.columns
                return pd.DataFrame({i: np.nan for i in col}, index=[dataframe.index[-1]])
            warnings.filterwarnings('ignore')
            numeric_df = dataframe._get_numeric_data()
            if min_periods is None:
                min_periods = 1
            correl=numeric_df.ewm(halflife=halflife, min_periods=min_periods).mean().iloc[-1,:].T
            if not isinstance(correl, pd.DataFrame):
                correl=correl.to_frame().T
            return correl


        def ewm_mean(dataframe: pd.DataFrame, halflife: int = 1, min_periods: int = 1, window: int = 1, verbose=True):
            from tqdm import trange # trange 等价于 tqdm(range())
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

        data = feather.read_dataframe(self.tmp_path + '\\ssf_v2中间数据.feather')
        # data = feather.read_dataframe(r'C:\Users\admin\Desktop\tmp.feather')
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
        # todo:
        if os.path.exists(self.savepath + f'\\SSF_v2_half{half}.feather'):
            dz = pd.read_feather(self.savepath + f'\\SSF_v2_half{half}.feather')
            dz = dz[dz['DATE'] <= self.end]
            dtls = dz['DATE'].unique()
            dtls.sort()
            data = data[data['DATE'] > min(self.end, dtls[max(len(dtls) - 66, 0)])]
        data = data.reset_index(drop=True)

        def getmean(x): # 一天之内所有股票的msf均值
            x['abs'] = abs(x['msf'] - x['msf'].mean())  # 适度日耀眼波动率
            # x['abs'] = (x['msf'] - x['msf'].mean()) ** 2
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
        data = pd.merge(ewm_std, pz1, how='left', on=['TICKER', 'DATE'])
        data[f'MSF_{half}'] = (data['mean_Vol_daily'] + data['std_Vol_daily']) / 2
        data = data.reset_index(drop=True)
        data = data[['TICKER', 'DATE', f'MSF_{half}']].reset_index(drop=True)
        # print(data)
        # feather.write_dataframe(data,r'C:\Users\admin\Desktop\test.feather')
        # return data
        if os.path.exists(self.savepath + f'\\SSF_v2_half{half}.feather'):
            dtz = pd.read_feather(self.savepath + f'\\SSF_v2_half{half}.feather')
            dtz = dtz[dtz['DATE'] <= self.end].reset_index(drop=True)
            # dtdate = dtz['DATE'].unique()
            # dtdate.sort()
            data = data.reset_index(drop=True)
            # data_ = data[data['DATE'] >= dtdate[-20]]
            # dtzz = dtz[dtz['DATE'] < dtdate[-20]]
            data_ = data[~data['DATE'].isin(dtz.DATE.unique())]
            data_ = pd.concat([dtz, data_])
            print(data_)
            feather.write_dataframe(data_, self.savepath + f'\\SSF_v2_half{half}.feather')
            feather.write_dataframe(data_, DataPath.factor_out_path + f'\\SSF_v2_half{half}.feather')

        else:
            feather.write_dataframe(data, DataPath.save_path_old + f'\\SSF_v2_half{half}.feather')
            feather.write_dataframe(data,DataPath.save_path_update + f'\\SSF_v2_half{half}.feather')
            print(data)
        print(f'SSF_v2_half{half} has been saved')
        print(np.setdiff1d(data.columns, ['TICKER', 'DATE']), sorted(data['DATE'].unique())[-1])

    def run(self):
        self.__daily_calcuation()
        df=self.cal_MSF_half()
        return df

def update(today='20250905'):
    MSF_object = MSF_half(start='20200101', end=today,savepath=DataPath.save_path_update)
    MSF_object.run()


if __name__ == '__main__':
    update()
