# @Author: Yixin Tian
# @File: auc_msf(已改).py
# @Date: 2025/11/18 13:22
# @Software: PyCharm
import os
import feather
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from tqdm import tqdm
from tool_tyx.path_data import DataPath


class MSF_half(object):
    def __init__(self, start='20220104', end='20251122', df_path=None, savepath=None, halflife=10,split_sec=1):
        self.halflife = halflife
        self.start = start
        self.end = end
        self.df_path=df_path
        self.path = DataPath.to_path
        self.savepath = savepath
        self.tmp_path=DataPath.tmp_path
        self.daily_list_file = os.listdir(DataPath.sh_min)
        self.split_sec=split_sec
        calendar = pd.read_csv(os.path.join(self.path,'calendar.csv'), dtype={'trade_date': str})
        calendar.rename(columns={'trade_date':'DATE'},inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']

    def __auc_prepare(self,date):
        df=feather.read_dataframe(os.path.join(self.df_path,f'{date}.feather'))
        df = df[(df['tradetime'] >= 92000000)&(df['tradetime'] < 92459950)]
        df['sec'] = df['tradetime']//1000
        seclist = df['sec'].unique().tolist()
        seclist.sort()
        stampmap = {}
        for x in range(0, len(seclist), self.split_sec):
            for j in range(self.split_sec):
                stampmap[seclist[min(x + j, len(seclist) - 1)]] = x // self.split_sec
        df['time_range'] = df['sec'].map(stampmap)
        df=df.sort_values(by=['TICKER','ApplSeqNum']).reset_index(drop=True)
        split_by_sec = df.groupby(['TICKER','time_range']).agg({'当前撮合成交量':['first','last'],'Price':['first','last','max','min']}).reset_index()
        split_by_sec.columns=['TICKER','time','vol_first','vol_last','open','close','high','low']
        split_by_sec['volume_diff']=split_by_sec['vol_last']-split_by_sec['vol_first']
        # split_by_sec.reset_index(inplace=True)
        split_by_sec['DATE']=date
        split_by_sec.drop(columns=['vol_first', 'vol_last'],inplace=True)
        # split_by_sec.rename(columns={'sec':'time','volume_diff':'volume'},inplace=True)
        # split_by_sec.rename(columns={'volume_diff':'volume'},inplace=True)
        return split_by_sec

    def __daily_calcuation(self):
        if os.path.exists(self.tmp_path + '\\auc_msf_basic_data.feather'):
            df_old = feather.read_dataframe(self.tmp_path + '\\auc_msf_basic_data.feather')
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
            threshold = xdif.mean() + xdif.std(),
            x = x.iloc[1:, :] # 不要932的数据，因为xdif retdiff值为nan
            x['retdiff'], x['mvdif'] = retdiff, xdif
            x = x.reset_index(drop=True)
            g, res = x[x['mvdif'] > threshold].index, []
            for i in g:
                if i < x.shape[0] - 5: # 不要最后5行，5分钟？
                    res.append(x['retdiff'].iloc[i:i + 5].std()) # 耀眼波动率：取往后5行retdiff的标准差
            a = sum(res) / len(res) if res else 0 #耀眼波动率的均值，日耀眼波动率
            return [x['TICKER'].unique()[0],x['DATE'].unique()[0], a]

        def get_daily(date):
            warnings.filterwarnings('ignore')
            filename = date + '.feather'
            if not os.path.exists(os.path.join(DataPath.sh_min,filename)):
                if date <= max_min_date:
                    raise FileNotFoundError
            else:
                to_day = np.array(['TICKER', 'DATE', 'msf'])
                dt=self.__auc_prepare(date).reset_index(drop=True)#[['TICKER','DATE','time','volume','open','close']]
                # print(dt)
                dt.rename(columns={'volume_diff':'volume'},inplace=True)
                dt = dt.drop(index=dt[np.isclose(dt['open'], 0)].index)
                dt = dt.dropna()
                dt['ret'] = dt['close'] / dt['open'] - 1
                # dt = dt.sort_values(by=['TICKER', 'min'])
                dt = dt.sort_values(by=['TICKER', 'time'])
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
            # *****************
        newdata=Parallel(n_jobs=3)(delayed(get_daily)(i) for i in tqdm(date_list))
        if newdata:
            to_day = pd.concat(newdata).reset_index(drop=True)
            if not df_old.empty:
                df_old['msf'] = df_old['msf'].astype(float)
            df_old = pd.concat([df_old, to_day]).sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
            df_old.to_feather(self.tmp_path + '\\auc_msf_basic_data.feather')

    def cal_MSF_half(self):
        data = feather.read_dataframe(os.path.join(self.tmp_path,'auc_msf_basic_data.feather'))
        rescols = list(np.setdiff1d(data.columns, ['TICKER', 'DATE']))
        for c in tqdm(rescols):
            tmp = data[['TICKER', 'DATE', c]]
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp[c] = tmp[c].fillna(tmp.groupby('DATE')[c].transform('median'))
            feather.write_dataframe(tmp, os.path.join(self.savepath, c + '.feather'))

    def run(self):
        self.__daily_calcuation()
        self.cal_MSF_half()


if __name__ == '__main__':
    # todo: 除了下面两个，其他都是默认参数，待改
    MSF_object = MSF_half(df_path=r'C:\Users\admin\Desktop',
                          savepath=r'C:\Users\admin\Desktop')
    MSF_object.run()
