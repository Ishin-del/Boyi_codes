import feather
import pandas as pd
import numpy as np
from joblib import Parallel,delayed
import os
import warnings
from tqdm import tqdm

"""
中信建投：投资者有限关注及注意力捕捉与溢出
"""
def process_na_stock(df,col):
    '''解决某些股票出现时间序列断开的情况，这种情况会在rolling（window!=min_period）的时候,因子值出现不同'''
    '''重组再上市情况'''
    pt = df.pivot_table(columns='TICKER', index='DATE', values=col)
    pt = pt.ffill()
    pp = pt.melt(ignore_index=False).reset_index(drop=False)
    df = pd.merge(pp, df, how='left', on=['TICKER', 'DATE'])
    use_cols = list(np.setdiff1d(df.columns, ['TICKER', 'DATE']))
    for c in use_cols:
        df[c] = df.groupby('TICKER')[c].ffill()
    df.drop(columns='value', inplace=True)
    return df

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


class Attention:
    def __init__(self,start,end,df_path,save_path,split_sec=1):
        self.start=start
        self.end=end
        self.df_path=df_path
        self.save_path=save_path
        self.split_sec=split_sec
        calendar = pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'), dtype={'trade_date': str})
        self.tar_lst = calendar[(calendar['trade_date'] >= self.start) & (calendar['trade_date'] <= self.end)]['trade_date']


    def __daily_calculation(self):
        warnings.filterwarnings('ignore')
        if os.path.exists(os.path.join(self.save_path,'high_freq.feather')):
            df_old=feather.read_dataframe(os.path.join(self.save_path,'high_freq.feather'))
            exist_date=df_old['DATE']['DATE'].unique().tolist()
        else:
            df_old=pd.DataFrame()
            exist_date=[]
        max_min_date=max(os.listdir(self.df_path))[:8] #订单薄路径下的最大日期
        date_list=np.setdiff1d(self.tar_lst,exist_date)
        # ---------------------------------------------------------------------
        def cal_fac(group):
            group.sort_values(by='float_mv', inplace=True)
            group.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp1 = group.iloc[:int(len(group) * 0.3)]
            tmp1['att_overflow'] = tmp1['attn'] - tmp1['attn'].mean()

            tmp2 = group.iloc[int(len(group) * 0.3):int(len(group) * 0.7)]
            tmp2['att_overflow'] = tmp2['attn'] - tmp2['attn'].mean()

            tmp3 = group.iloc[int(len(group) * 0.7):len(group)]
            tmp3['att_overflow'] = tmp3['attn'] - tmp3['attn'].mean()

            tmp = pd.concat([tmp1, tmp2, tmp3])
            # print(tmp)
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            return tmp

        def get_date(tmp):
            warnings.filterwarnings('ignore')
            df = tmp.groupby('citic_code', as_index=False, group_keys=False).apply(cal_fac)
            # print(df)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            return df

        def get_daily(date):
            warnings.filterwarnings('ignore')
            if not os.path.exists(os.path.join(self.df_path,date+'.feather')):
                if date <= max_min_date:
                    raise FileNotFoundError
            else:
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
                split_by_sec = dt.groupby('TICKER').agg({'当前撮合成交量': ['first', 'last'], 'Price': ['first', 'last',
                               'max', 'min']}).reset_index()
                split_by_sec.columns = ['TICKER', 'vol_first', 'vol_last', 'open', 'close', 'high', 'low']
                split_by_sec['volume_diff'] = split_by_sec['vol_last'] - split_by_sec['vol_first']
                split_by_sec['DATE'] = date
                split_by_sec.drop(columns=['vol_first', 'vol_last'], inplace=True)
                dt = split_by_sec.reset_index(drop=True)
                return dt
        # ---------------------------------------------------------------------

        ind_data=feather.read_dataframe(DataPath.ind_df_path)
        ind_data=ind_data[(ind_data.DATE>=self.start)&(ind_data.DATE<=self.end)]

        # daily_df=feather.read_dataframe(DataPath.daily_path,columns=['DATE','TICKER','close'])
        # daily_df=daily_df[(daily_df.DATE>=self.start)&(daily_df.DATE<=self.end)]
        daily_df=Parallel(n_jobs=3)(delayed(get_daily)(date) for date in tqdm(self.tar_lst,desc='竞价日频数据合成中'))
        daily_df=pd.concat(daily_df).reset_index(drop=True)
        daily_df.sort_values(['TICKER','DATE'],inplace=True)
        daily_df['rtn']=daily_df.groupby('TICKER')['close'].pct_change()
        daily_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        daily_df['rtn_mean']=daily_df.groupby('DATE')['rtn'].transform('mean')
        daily_df['ab_norm_ret']=np.square(daily_df['rtn']-daily_df['rtn_mean'])
        daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
        daily_df=process_na_stock(daily_df,'close')
        # daily_df['attn']=daily_df.groupby('TICKER')['ab_norm_ret'].rolling(21,5).mean().values#.dropna(how='all')
        daily_df['attn']=daily_df['ab_norm_ret']
        daily_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        # daily_df.dropna(axis=0,how='any',inplace=True)

        market_cap=feather.read_dataframe(os.path.join(DataPath.to_df_path,'float_mv.feather'))
        market_cap=market_cap[(market_cap.DATE>=self.start)&(market_cap.DATE<=self.end)]
        daily_df=daily_df.merge(ind_data,on=['TICKER','DATE'],how='inner').merge(market_cap,on=['TICKER','DATE'],how='inner')
        # print(daily_df)
        daily_df.replace([np.inf,-np.inf],np.nan,inplace=True)
        daily_df=process_na_stock(daily_df,'close')
        daily_df.rename(columns={'CITIC_CODE':'citic_code'},inplace=True)
        daily_df=daily_df[['DATE','TICKER','attn','citic_code', 'float_mv']]
        daily_df.sort_values(['TICKER','DATE'],inplace=True)
        daily_df[['citic_code','float_mv']]=daily_df.groupby('TICKER')[['citic_code','float_mv']].shift(1)
        # print(daily_df)
        res= Parallel(n_jobs=10)(delayed(get_date)(daily_df[daily_df.DATE==date]) for date in tqdm(self.tar_lst))
        # for date in tqdm(tar_date_list):
        #     res=cal_factors1(daily_df[daily_df.DATE==date])
        #     print(res)
        #     break
        # print(res)
        res=pd.concat(res).reset_index(drop=True)[['DATE','TICKER','att_overflow']]
        res.rename(columns={'att_overflow':'auc_att_overflow'},inplace=True)
        res.replace([np.inf,-np.inf],np.nan,inplace=True)
        res['auc_att_overflow']=res['auc_att_overflow'].fillna(res.groupby('DATE')['auc_att_overflow'].transform('median'))
        res.dropna(inplace=True)
        res=pd.concat([df_old,res]).sort_values(by=['TICKER','DATE']).reset_index(drop=True)
        feather.write_dataframe(res,os.path.join(self.save_path,'auc_att_overflow.feather'))

        print(res)

    def run(self):
        self.__daily_calculation()

if  __name__=='__main__':
    obj=Attention('20220104','20220404',
                  df_path=r'\\192.168.1.211\d\zkh\Price_and_vols_at_auction_time\concat_daily',
                  save_path=r'C:\Users\admin\Desktop')
    obj.run()