import os
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings


class DataPath():
    # 数据存储路径-----------------------------------------------------------------
    factor_out_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子'
    save_path_old=r'D:\tyx\检查更新\2020-2022.12'
    save_path_update=r'D:\tyx\检查更新\2021.6-2025.8'
    save_help_path=r'D:\tyx\raw_data'
    tmp_path=r'D:\tyx\中间数据'
    # tmp_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用'

    # 数据路径-------------------------------------------------------
    # daily_path=r'\\192.168.1.101\local_data\Data_Storage\daily.feather'
    daily_path=r'Z:\local_data\Data_Storage\daily.feather'
    # sh_min=r'\\DESKTOP-79NUE61\SH_min_data' # 上海数据 逐笔合分钟数据
    sh_min=r'\\192.168.1.210\SH_min_data' # 上海数据 逐笔合分钟数据
    sz_min=r'\\DESKTOP-79NUE61\SZ_min_data' # 深圳数据 逐笔合分钟数据
    feather_2022 = r'\\192.168.1.28\h\data_feather(2022)\data_feather'
    feather_2023 = r'\\192.168.1.28\h\data_feather(2023)\data_feather'
    feather_2024 = r'\\192.168.1.28\i\data_feather(2024)\data_feather'
    feather_2025 = r'\\192.168.1.28\i\data_feather(2025)\data_feather'
    moneyflow_sh=r'\\DESKTOP-79NUE61\money_flow_sh'
    moneyflow_sz=r'\\DESKTOP-79NUE61\money_flow_sz'
    # moneyflow数据按照4万，20万，100万为分界线，分small,medium,large,xlarge,此数据将集合竞价数据考虑进来了
    # mkt_index=r'\\192.168.1.101\local_data\ex_index_market_data\day'
    # to_df_path=r'\\192.168.1.101\local_data\Data_Storage' #float_mv.feather,adj_factors(复权因子)数据路径,citic_code.feather数据(行业数据),money_flow
    # to_data_path=r'\\192.168.1.101\local_data\base_data' #totalShares数据路径
    # to_path=r'\\192.168.1.101\local_data' #calendar.csv数据路径
    # wind_A_path=r'\\192.168.1.101\ssd\local_data\Data_Storage\881001.WI.csv' #万得全A指数路径
    # ind_df_path = r'\\192.168.1.101\local_data\Data_Storage\citic_code.feather'
    mkt_index = r'Z:\local_data\ex_index_market_data\day'
    to_df_path = r'Z:\local_data\Data_Storage'  # float_mv.feather,adj_factors(复权因子)数据路径,citic_code.feather数据(行业数据),money_flow
    to_data_path = r'Z:\local_data\base_data'  # totalShares数据路径
    to_path = r'Z:\local_data'  # calendar.csv数据路径
    wind_A_path = r'Z:\local_data\Data_Storage\881001.WI.csv'  # 万得全A指数路径
    ind_df_path = r'Z:\local_data\Data_Storage\citic_code.feather'

    # feather_sh=r'\\Desktop-79nue61\sh'
    # feather_sz=r'\\Desktop-79nue61\sz'
    feather_sh=r'\\192.168.1.210\sh'
    feather_sz=r'\\192.168.1.210\sz'

    order_min_sh=r'\\DESKTOP-79NUE61\SH_min_data_big_order'
    order_min_sz=r'\\DESKTOP-79NUE61\SZ_min_data_big_order'
    # 财务数据路径
    financial_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\财务数据'

    # -------------------------------------------
    # 机器学习数据路径
    train_data_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\rolling_generation_202508_sz_auction_end_20240815_908\train'
    train_big_order_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\rolling_generation_202508_sz_auction_end_20240815_909_big_order\train'

    # 最终因子路径
    factor_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子'
    ret_df_path = r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\return_因子组合用.feather'


    snap_path_2022=fr'\\192.168.1.7\data01\data_feather'
    snap_path_2023=fr'\\192.168.1.7\data02\data_feather'
    snap_path_2024=fr'\\192.168.1.7\data03\data_feather'
    snap_path_2025=fr'\\192.168.1.7\data04\data_feather'


class MSF_half(object):
    def __init__(self, start, end, df_path,halflife=10,savepath=None,split_sec=1):
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
        # todo:
        if os.path.exists(self.tmp_path + '\\auc_ssf_v2中间数据.feather'):
            df_old = feather.read_dataframe(self.tmp_path + '\\auc_ssf_v2中间数据.feather')
            exist_date = []
            for i in df_old['DATE'].unique():
                exist_date.append(''.join(i.split('-')))
        else:
            df_old = pd.DataFrame()
            exist_date = []
        max_min_date = max(self.daily_list_file).replace('.feather', '')
        date_list = np.setdiff1d(self.daily_list, exist_date)
        # date_list=['20220104.feather']
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
            if not os.path.exists(os.path.join(DataPath.sh_min,filename)) and False:
                if date <= max_min_date:
                    raise FileNotFoundError
            else:
                to_day = np.array(['TICKER', 'DATE', 'msf'])
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
                dt=split_by_sec.reset_index(drop=True)
                # -------------------------------------------------------------------
                dt.rename(columns={'volume_diff':'volume'},inplace=True)
                dt = dt.drop(index=dt[np.isclose(dt['open'], 0)].index)
                dt = dt.dropna()
                dt['ret'] = dt['close'] / dt['open'] - 1
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
        newdata=Parallel(n_jobs=14)(delayed(get_daily)(i) for i in tqdm(date_list))
        print(newdata)
        if newdata:
            to_day = pd.concat(newdata).reset_index(drop=True)
            if not df_old.empty:
                df_old['msf'] = df_old['msf'].astype(float)
            df_old = pd.concat([df_old, to_day]).sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
            df_old.to_feather(self.tmp_path + '\\auc_ssf_v2中间数据.feather')

    def cal_SSF(self):
        data = feather.read_dataframe(os.path.join(self.tmp_path, 'auc_ssf_v2中间数据.feather'))
        rescols = list(np.setdiff1d(data.columns, ['TICKER', 'DATE']))
        for c in tqdm(rescols):
            c='auc_'+c
            tmp = data[['TICKER', 'DATE', c]]
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp[c] = tmp[c].fillna(tmp.groupby('DATE')[c].transform('median'))
            feather.write_dataframe(tmp, os.path.join(self.savepath, c + '.feather'))

    def run(self):
        self.__daily_calcuation()
        df=self.cal_SSF()
        return df


if __name__ == '__main__':
    MSF_object = MSF_half(start='20220104', end='20220104',
                          df_path=r'C:\Users\admin\Desktop',
                          savepath=r'C:\Users\admin\Desktop')
    MSF_object.run()
