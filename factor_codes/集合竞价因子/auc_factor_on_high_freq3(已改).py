import os
import warnings
import feather
import numpy as np
import pandas as pd
from tqdm import tqdm


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


def Garman_Klass(x):
    num1=np.log(x['high'] / x['low']) ** 2
    num2=np.log(x['close'] / x['open']) ** 2
    num3=2 * np.log(2) - 1
    return (0.5 *num1 - num3 * num2).mean()

def Rogers_Satchell(x):
    num1=np.log(x['high'] / x['close'])
    num2=np.log(x['high'] / x['open'])
    num3=np.log(x['low'] / x['close'])
    num4=np.log(x['low'] / x['open'])
    return (num1*num2+num3*num4).mean()

def cho_free(x):
    num=np.log(1+1/8)
    miu=x['pre_log_close'].mean()/2*num
    try:
        tao=x['pre_log_close'].mean()/len(x)
    except ZeroDivisionError:
        return np.nan
    return 2*miu*num/(np.log(num+miu*tao)-np.log(num-miu*tao))

def yang_zhang(x,sigma_RS):
    # todo: alph=0.34
    sigma_o=np.abs(x['log_open']-x['pre_log_close']-1).sum()
    try:
        k=(0.34/(1.34+(len(x)+1)/(len(x)-1)))
    except ZeroDivisionError:
        return np.nan
    sigma_c=np.abs(x['log_close']-x['pre_log_open']-1).sum()
    # sigma_RS=(x['sigma_RS'].iloc[0])**2
    sigma_RS=sigma_RS**2
    return sigma_o+k*sigma_c+(1-k)*sigma_RS

def EDGE(group,pi,pi_c):
    # todo:这个应该有点不对
    # print(pi)
    # pi=pi[pi.index==group['TICKER'].iloc[0]].iloc[0]
    # pi_c=pi_c[pi_c.index==group['TICKER'].iloc[0]].iloc[0]
    x1=pi/2*(group['midd_price'] - group['log_open']) * (group['log_open'] - group['pre_midd_price'])+pi_c/2*(group['midd_price'] - group['pre_log_close']) * (group['pre_log_close'] - group['pre_midd_price'])
    x2=pi/2*(group['midd_price'] - group['log_open']) * (group['log_open'] - group['pre_log_close'])+pi_c/2*((group['log_open'] - group['pre_log_close']) * (group['pre_log_close'] - group['pre_midd_price']))
    try:
        w1=np.var(x2)/(np.var(x1)+np.var(x2))
        w2=np.var(x1)/(np.var(x1)+np.var(x2))
    except ZeroDivisionError:
        return np.nan
    return w1*np.mean(x1)+w2*np.mean(x2)

def OHL(group): # OHL
    tmp1 = ((group['midd_price'] - group['log_open']) * (group['log_open'] - group['pre_midd_price'])).mean()
    tmp2 = (group['midd_price'] - group['log_open']).mean()
    tmp3 = ((group['log_open'] - group['pre_midd_price']) * group['tau']).mean()
    tau_mean = group['tau'].mean()
    return tmp1 - (tmp2 * tmp3) / tau_mean

def CHL(group): # CHL
    tmp1 = ((group['midd_price'] - group['pre_log_close']) * (group['pre_log_close'] - group['pre_midd_price'])).mean()
    tmp2 = (group['midd_price'] - group['pre_log_close']).mean()
    tmp3 = ((group['pre_log_close'] - group['pre_midd_price']) * group['tau']).mean()
    tau_mean = group['tau'].mean()
    return tmp1 - (tmp2 * tmp3) / tau_mean

def OHLC(group): #OHLC
    tmp1 = ((group['midd_price'] - group['log_open']) * (group['log_open'] - group['pre_log_close'])).mean()
    tmp2 = (group['midd_price'] - group['log_open']).mean()
    tmp3 = ((group['log_open'] - group['pre_log_close']) * group['tau']).mean()
    tau_mean = group['tau'].mean()
    return tmp1 - (tmp2 * tmp3) / tau_mean

def CHLO(group): #CHLO
    tmp1 = ((group['log_open'] - group['pre_log_close']) * (group['pre_log_close'] - group['pre_midd_price'])).mean()
    tmp2 = (group['log_open'] - group['pre_log_close']).mean()
    tmp3 = ((group['pre_log_close'] - group['pre_midd_price']) * group['tau']).mean()
    tau_mean = group['tau'].mean()
    return tmp1 - (tmp2 * tmp3) / tau_mean

class auc_high_freq:
    def __init__(self,start='20220101',end='20251119',data_path=None,save_path=None,tmp_path=None,split_sec=1):
        self.start=start
        self.end=end
        self.save_path=save_path
        self.data_path = data_path
        self.tmp_path = tmp_path
        self.split_sec = split_sec
        self.daily_list_file = os.listdir(self.data_path)
        calendar=pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'),dtype={'trade_date': str})
        self.daily_lst=calendar[(calendar['trade_date'] >= self.start) & (calendar['trade_date'] <= self.end)]['trade_date']

    def __daily_calculation(self):
        if os.path.exists(os.path.join(self.tmp_path,'high_freq.feather')):
            df_old=feather.read_dataframe(os.path.join(self.tmp_path,'high_freq.feather'))
            exist_date=df_old['DATE']['DATE'].unique().tolist()
        else:
            df_old=pd.DataFrame()
            exist_date=[]
        max_min_date=max(self.daily_list_file)[:8] #订单薄路径下的最大日期
        date_list=np.setdiff1d(self.daily_lst,exist_date) # 检查更新要进行计算的日期，或生成新因子要计算的日期
        fct_lst=['ROLL','CS','AR','OHL','Parkinson','Garman_Klass','Rogers_Satchell','Yang_Zhang','Cho_Frees','CHL','OHLC','CHLO','EDGE']
        self.fct_lst=fct_lst
        data_path=self.data_path

        def get_date(x):
            warnings.filterwarnings('ignore')
            # 因子计算逻辑
            for col in ['close', 'open', 'high', 'low']:
                x[f'log_{col}'] = np.log(x[col])
            x.sort_values(['TICKER', 'time'], inplace=True)
            x['pre_log_close'] = x['log_close'].shift(1)
            x['pre_log_open'] = x['log_open'].shift(1)
            x['log_close_diff'] = x['log_close'].diff()
            x['pre_log_close_diff'] = x['pre_log_close'].diff()
            x.replace([np.inf, -np.inf], np.nan, inplace=True)
            x.dropna(inplace=True)
            # print(x.columns)
            # Roll价差
            res1 = np.sqrt(np.maximum(-4 * np.cov(x['log_close_diff'], x['pre_log_close_diff'])[0, 1], 0))
            # CS价差
            x['log_high_low'] = np.log(x['high'] / x['low'])
            x['pre_log_high_low'] = x['log_high_low'].shift(1)
            x['pre_high'] = x['high'].shift(1)
            x['pre_low'] = x['low'].shift(1)
            x['max_high'] = x[['high', 'low', 'pre_high', 'pre_low']].max(axis=1)
            x['min_low'] = x[['high', 'low', 'pre_high', 'pre_low']].min(axis=1)
            x['gamma'] = (np.log(x['max_high'] / x['min_low'])) ** 2
            x['beta'] = ((x['log_high_low']) ** 2 + (x['pre_log_high_low']) ** 2) / 2
            x['alpha'] = (np.sqrt(2 * x['beta']) - np.sqrt(x['beta'])) / (3 - 2 * np.sqrt(2)) - np.sqrt(
                x['gamma'] / (3 - 2 * np.sqrt(2)))
            x['CS'] = 2 * (np.exp(x['alpha']) - 1) / (1 + np.exp(x['alpha']))
            x.replace([np.inf, -np.inf], np.nan, inplace=True)
            x.dropna(inplace=True)
            res2 = x['CS'].mean()
            # AR
            x['midd_price'] = (x['log_high'] + x['log_low']) / 2
            x.sort_values(['TICKER', 'time'], inplace=True)
            x['pre_close'] = x['close'].shift(1)
            x['pre_open'] = x['open'].shift(1)
            # x['close_diff']=x['close'].diff()
            x['pre_midd_price'] = x['midd_price'].shift(1)
            x['pre_log_close'] = x['log_close'].shift(1)
            x['AR'] = np.sqrt(
                np.maximum(-4 * (x['pre_log_close'] - x['pre_midd_price']) * (x['pre_log_close'] - x['midd_price']), 0))
            x.replace([np.inf, -np.inf], np.nan, inplace=True)
            x.dropna(inplace=True)
            res3 = x['AR'].mean()

            x['tau'] = np.where((x['log_high'] == x['log_low']) & (x['log_low'] == x['pre_log_close']), 0, 1)
            P1 =(x['log_open'] != x['log_high']).mean()
            P2 =(x['log_open'] != x['log_low']).mean()
            P1_c = (x['log_close'] != x['log_high']).mean()
            P2_c = (x['log_close'] != x['log_low']).mean()
            tau_mean = x['tau'].mean()
            P1 *= tau_mean
            P2 *= tau_mean
            P1_c *= tau_mean
            P2_c *= tau_mean
            pi, pi_c = -8.0 / (P1 + P2), -8.0 / (P1_c + P2_c)
            # OHL
            tmp_data = OHL(x)
            S_square_OHL = pi * tmp_data
            x.replace([np.inf, -np.inf], np.nan, inplace=True)
            x.dropna(inplace=True)
            res4 = np.sqrt(S_square_OHL) * np.where(S_square_OHL > 0, 1, -1)
            # CHL
            tmp_data = CHL(x)
            S_square_CHL = pi_c * tmp_data
            res10 = np.sqrt(S_square_CHL) * np.where(S_square_CHL > 0, 1, -1)
            # OHLC
            tmp_data = OHLC(x)
            S_square_OHLC = pi * tmp_data
            res11 = np.sqrt(S_square_OHLC) * np.where(S_square_OHLC > 0, 1, -1)
            # CHLO
            tmp_data = CHLO(x)
            S_square_CHLO = pi_c * tmp_data
            res12 = np.sqrt(S_square_CHLO) * np.where(S_square_OHLC > 0, 1, -1)
            # EDGE
            S_square_EDGE = EDGE(x, pi, pi_c)
            res13 = np.sqrt(S_square_EDGE) * np.where(S_square_EDGE > 0, 1, -1)
            # 波动率因子========================================================
            x.replace([np.inf, -np.inf], np.nan, inplace=True)
            x.dropna(inplace=True)
            res5 = np.sqrt(((np.log(x['high'] / x['low'])) ** 2).mean() * 1 / 4 * np.log(2))# Parkinson
            res6 = np.sqrt(Garman_Klass(x))  # Garman_Klass
            res7 = np.sqrt(Rogers_Satchell(x))  #
            # x = x.merge(res7.reset_index(), on='TICKER', how='left').rename(columns={0: 'sigma_RS'})
            res8 = np.sqrt(yang_zhang(x,res7))  # Yang-Zhang
            res9 = np.sqrt(cho_free(x))
            res = [res1, res2, res3, res4, res5, res6, res7, res8, res9, res10, res11, res12, res13]
            # res = pd.concat(res, axis=1).reset_index()
            # res['DATE'] = date
            return res


        def get_daily(date):
            warnings.filterwarnings('ignore')
            if not os.path.exists(os.path.join(data_path,date+'.feather')):
                pass
                if date <= max_min_date:
                    raise FileNotFoundError
            else:
                # todo：
                # to_day=np.array(['TICKER','DATE']+fct_lst)
                # -------------------------------------------------------------------
                dt = feather.read_dataframe(os.path.join(self.data_path, f'{date}.feather'))
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
                # -------------------------------------------------------------------
                dt=dt.groupby('TICKER').apply(get_date)
                dt = dt.reset_index()
                dt[fct_lst] = dt[0].apply(pd.Series)
                dt=dt.drop(columns=0)
                dt['DATE']=date
                return dt
        newdata = []
        for i in tqdm(date_list):
            newdata.append(get_daily(i))  # todo:可以parallel
        if newdata: # 不为空的话
            to_day=pd.concat(newdata).reset_index(drop=True)
            if not df_old.empty: # 因子更新
                for i in fct_lst:
                    df_old[i]=df_old[i].astype(float) # 已存在文件 因子列变成float格式
            df_old=pd.concat([df_old,to_day]).sort_values(['TICKER','DATE']).reset_index(drop=True)
            feather.write_dataframe(df_old,os.path.join(self.tmp_path,'high_freq.feather')) # 中间数据保存
            # return df_old

    def cal(self):
        warnings.filterwarnings('ignore')
        data = feather.read_dataframe(self.tmp_path+'\high_freq.feather')
        factors = np.setdiff1d(data.columns, ['TICKER', 'DATE'])
        for c in factors:
            # print()
            tmp=data[['TICKER', 'DATE', c]]
            tmp.replace([np.inf,-np.inf],np.nan,inplace=True)
            tmp[c]=tmp[c].fillna(tmp.groupby('DATE')[c].transform('median'))
            # print(tmp)
            feather.write_dataframe(tmp, self.savepath + f'{c}.feather')
    def run(self):
        # self.__daily_calculation()
        self.cal()

if __name__=='__main__':
    object = auc_high_freq('20220104','20220104',tmp_path=r'C:\Users\admin\Desktop',
                           save_path=r'C:\Users\admin\Desktop',
                           data_path=r'C:\Users\admin\Desktop')
    object.run()