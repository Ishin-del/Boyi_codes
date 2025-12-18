import os
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib.parallel import Parallel, delayed, cpu_count
import warnings

from tool_tyx.tyx_funcs import get_auc_data
from tool_tyx.path_data import DataPath


# 月耀眼收益率 SSF 版本2：半衰期合成
# 对应研报 ： 20220412-方正证券-多因子选股系列研究之一：成交量激增时刻蕴含的alpha信息


class SSF_half(object):
    def __init__(self, start='20250102', end='20250228', savepath=None, halflife=10,df_path=None,split_sec=1):
        self.split_sec = split_sec
        self.halflife = halflife
        self.start = start
        self.end = end
        self.tmp_path = DataPath.tmp_path
        self.df_path = df_path
        self.savepath = savepath
        self.daily_list_file = os.listdir(DataPath.sh_min)
        calendar = pd.read_csv(DataPath.to_path + '\\calendar.csv', dtype={'trade_date': str})
        calendar.rename(columns={'trade_date': 'DATE'}, inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']

    def __daily_calcuation(self):
        if os.path.exists(self.tmp_path + '\\auc_ssf_basic_data.feather'):
            df_old = feather.read_dataframe(self.tmp_path + '\\auc_ssf_basic_data.feather')
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []

        # max_min_date = max(self.daily_list_file)[:10].replace('-', '')
        max_min_date = max(self.daily_list_file).replace('.feather', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)
        # date_list=['20220104.feather']
        # **************    主要函数部分,获取日内的数据
        def get_date(x):
            # 1 求分钟交易额的差值
            x = x.sort_values(by=['time'])
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
            if not os.path.exists(os.path.join(DataPath.sh_min,filename)) and False:
                if date <= max_min_date:
                    raise FileNotFoundError
            else:
                to_day = np.array(['TICKER', 'DATE', 'ssf'])
                # -------------------------------------------------------------------
                dt = feather.read_dataframe(os.path.join(self.df_path, f'{date}.feather'))
                dt = dt[(dt['tradetime'] >= 92000000) & (dt['tradetime'] < 92459950)]
                dt['sec'] = dt['tradetime'] // 1000
                seclist = dt['sec'].unique().tolist()
                seclist.sort()
                stampmap = {}
                for x in range(0, len(seclist), self.split_sec):
                    for j in range(self.split_sec):
                        stampmap[seclist[min(x + j, len(seclist) - 1)]] = x // self.split_sec
                dt['time_range'] = dt['sec'].map(stampmap)
                dt = dt.sort_values(by=['TICKER', 'ApplSeqNum']).reset_index(drop=True)
                split_by_sec = dt.groupby(['TICKER', 'time_range']).agg(
                    {'当前撮合成交量': ['first', 'last'], 'Price': ['first', 'last', 'max', 'min']}).reset_index()
                split_by_sec.columns = ['TICKER', 'time', 'vol_first', 'vol_last', 'open', 'close', 'high', 'low']
                split_by_sec['volume_diff'] = split_by_sec['vol_last'] - split_by_sec['vol_first']
                split_by_sec['DATE'] = date
                split_by_sec.drop(columns=['vol_first', 'vol_last'], inplace=True)
                dt = split_by_sec.reset_index(drop=True)
                dt.rename(columns={'volume_diff':'volume'},inplace=True)
                # ---------------------------------------------------
                # dt.rename(columns={'time':'min'},inplace=True)
                # dt=dt[['TICKER', 'DATE', 'min', 'volume', 'open', 'close']]
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
                # print(to_day)
                return to_day
        newdata = Parallel(n_jobs=10)(delayed(get_daily)(i) for i in tqdm(date_list,desc='calculating daily'))
        if len(newdata) > 0:
            to_day = pd.concat(newdata)
            if not df_old.empty:
                df_old['ssf'] = df_old['ssf'].astype(float)
            df_old = pd.concat([df_old, to_day]).sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
            df_old.rename(columns={'ssf':'auc_ssd'},inplace=True)
            # print(df_old)
            df_old.to_feather(self.tmp_path + '\\auc_ssf_basic_data.feather')

    def cal_SSF(self):
        data = feather.read_dataframe(os.path.join(self.tmp_path, 'auc_msf_basic_data.feather'))
        data.rename(columns={'ssf':'auc_ssf'},inplace=True)
        print(data)
        rescols = list(np.setdiff1d(data.columns, ['TICKER', 'DATE']))
        for c in tqdm(rescols):
            tmp = data[['TICKER', 'DATE', c]]
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp[c] = tmp[c].fillna(tmp.groupby('DATE')[c].transform('median'))
            feather.write_dataframe(tmp, os.path.join(self.savepath, c + '.feather'))

    def run(self):
        """
        方便外部调用
        在统一的全部因子更新脚本中，对于每个import的类都会只运行run这个方法，以实现因子更新的目的
        """
        self.__daily_calcuation()
        self.cal_SSF()

if __name__ == '__main__':
    # update()
    # todo: 参数待改
    SSF_half(start='20220104', end='20220104',
             df_path=r'C:\Users\admin\Desktop',
             savepath=r'C:\Users\admin\Desktop').run()