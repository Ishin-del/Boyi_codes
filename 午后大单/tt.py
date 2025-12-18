
import warnings
from itertools import chain
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from tqdm import tqdm
import os
import feather


class Big_Order:
    def __init__(self, start, end, calender_path=None, df_path=None, savepath=None):
        self.start = start or '20220104'
        self.end = end or '20251121'
        self.df_path = df_path or r'X:\data_feather'
        self.calender_path = calender_path or r'\\192.168.1.210\Data_Storage2'
        self.savepath = savepath or rf'D:\因子保存\{self.__class__.__name__}'
        self.tmp_path = os.path.join(self.savepath, 'basic')

        os.makedirs(self.savepath, exist_ok=True)
        os.makedirs(str(self.tmp_path), exist_ok=True)

        self.daily_list_file = os.listdir(r'X:\data_feather')
        calendar = pd.read_csv(os.path.join(self.calender_path, 'calendar.csv'), dtype={'trade_date': str})
        calendar.rename(columns={'trade_date': 'DATE'}, inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']
        self.threshold_df=feather.read_dataframe(r'\\192.168.1.210\Factor_Storage\田逸心\calc6临时用\大单阈值.feather')

    def __daily_calculation(self):
        # if os.path.exists(os.path.join(self.tmp_path, 'high_freq_skew.feather')):
        #     df_old = feather.read_dataframe(os.path.join(self.tmp_path, 'high_freq_skew.feather'))
        #     exist_date = []
        #     for i in df_old['DATE'].unique():
        #         exist_date.append(''.join(i.split('-')))
        # else:
        #     df_old = pd.DataFrame()
        #     exist_date = []
        # max_min_date = max(os.listdir(self.df_path))
        # date_list = np.setdiff1d(self.daily_list, exist_date)

        def get_code(date,code):
            warnings.filterwarnings('ignore')
            dt = feather.read_dataframe(os.path.join(self.df_path, fr'{date}\hq_trade_spot\{code}'))
            dt.drop(columns=['index', 'tradedate', 'ChannelNo', 'MDStreamID', 'SecurityIDSource'], inplace=True)
            dt.sort_values('ApplSeqNum', inplace=True)
            tmp = dt[(dt['tradetime'] > int(f'{date}092500000')) & (dt['tradetime'] <= int(f'{date}113000000'))]
            qty_11 = tmp[tmp['ExecType'] == 'F']['TradeQty'].sum()
            dt = dt[(dt['tradetime'] >= int(f'{date}130000000')) & (
                        dt['tradetime'] <= int(f'{date}130000000') + 6000)].reset_index(drop=True)
            # ------------------------------------
            # 计算1s内的委托量，并保证 委托的价格大于等于最近一笔的成交价（买方）
            dt['最近一笔成交价'] = np.where(dt['ExecType'] == 'F', dt['Price'], np.nan)  # 如果成交用当前价格，否则填nan值，然后用前面数据填充
            dt['最近一笔成交价'] = dt['最近一笔成交价'].ffill()
            tar_df = dt[(dt['tradetime'] <= int(f'{date}130001000')) & (dt['BidApplSeqNum'] > dt['OfferApplSeqNum']) &
                        (dt['Price'] >= dt['最近一笔成交价'])]
            tar_df_num = tar_df.groupby('BidApplSeqNum').agg({'ApplSeqNum': 'last'})  # 一笔订单分几波成交，按最后成交结束对应的时间来定格
            tar_df = tar_df.groupby('BidApplSeqNum').agg({'TradeQty': 'sum'}).rename(
                columns={'TradeQty': 'TradeQty_sum'})
            tar_df = tar_df.merge(tar_df_num, right_index=True, left_index=True, how='outer')
            code = code.replace('.feather', '.SZ')
            vols_threshold=self.threshold_df[(self.threshold_df['DATE']==date)&(self.threshold_df['TICKER']==code)]
            if vols_threshold.empty:
                return
            vols_threshold=vols_threshold['vols_threshold'].iloc[0]*0.4
            tar_df = tar_df[tar_df['TradeQty_sum'] >= vols_threshold]  # tar_df选出大于等于阈值的行
            if tar_df.empty:
                return
            dt1 = dt[(dt['BidApplSeqNum'] > dt['OfferApplSeqNum']) & (dt['ExecType'] == 'F')]  # 只保留buy的数据 和 成交数据
            res = []
            for _, row in tar_df.iterrows():
                big_order_qty=tar_df['TradeQty_sum'].iloc[0]
                # 1s内计算主动成交量，主动成交笔数
                cal_df = dt1[dt1['ApplSeqNum'] > row['ApplSeqNum']]
                if cal_df.empty:
                    continue
                cal_df = cal_df[cal_df['tradetime'] <= cal_df['tradetime'].iloc[0] + 1000]
                if cal_df.empty:
                    continue
                act_volume = cal_df[cal_df['Price'] >= cal_df['最近一笔成交价']]['TradeQty'].sum()  # 主动成交量
                act_volume_num = len(cal_df['BidApplSeqNum'])  # 主动成交笔数
                qty11_to_qty13 =  act_volume/qty_11
                # ====================================================================
                # 5s内计算vwap
                dt2 = dt[(dt['ApplSeqNum'] > row['ApplSeqNum']) & (dt['ExecType'] == 'F')]
                the_time = dt[dt['ApplSeqNum'] > row['ApplSeqNum']]['tradetime'].iloc[0] + 5000  # 5s内
                dt2 = dt2[dt2['tradetime'] <= the_time]
                vwap_5s = (dt2['Price'] * dt2['TradeQty']).sum() / dt2['TradeQty'].sum()
                res.append([code, date, big_order_qty,act_volume, act_volume_num, qty11_to_qty13, vwap_5s])
            print(res)
            return res

        def get_daily(date):
            tar_codes=os.listdir(os.path.join(self.df_path,fr'{date}\hq_trade_spot'))
            # total_res=[]
            # for code in tqdm(tar_code):
            #     total_res.append(get_code(date,code))
            Parallel(n_jobs=10)(delayed(get_code)(date,code) for code in tqdm(tar_codes) )
            # total_res = []
            # with ProcessPoolExecutor(max_workers=10) as executor:
            #     future_to_code = {executor.submit(get_code, date, code): code for code in tqdm(tar_codes)}
            #     for future in as_completed(future_to_code):
            #         total_res.append(future.result())
            # total_res=[x for x in total_res if x!=None]
            # df=pd.DataFrame(list(chain.from_iterable(total_res)))
            # df.columns=['TICKER','DATE','大单成交量','主动成交量','主动成交笔数','1点与上午成交量占比','vwap_5s']
            # return df

        df=get_daily('20250102')
        print(df)
        # df.to_csv(r'C:\Users\admin\Desktop\tt.csv')
        # df = Parallel(n_jobs=10)(delayed(get_daily)(date) for date in tqdm(date_list))
        # df = pd.concat(df).reset_index(drop=True)
        # alls = pd.concat([df_old, df])
        # alls =alls.drop_duplicates(subset=['TICKER','DATE'],keep='first').reset_index(drop=True)
        # feather.write_dataframe(alls, os.path.join(self.tmp_path, 'high_freq_skew.feather'))

    def cal(self):
        warnings.filterwarnings('ignore')
        df = feather.read_dataframe(os.path.join(self.tmp_path, 'high_freq_skew.feather'))
        tar_col = list(np.setdiff1d(df.columns, ['TICKER', 'DATE']))
        for c in tar_col:
            tmp = df[['TICKER', 'DATE', c]]
            tmp.rename(columns={c: 'auc_' + c}, inplace=True)
            c = 'auc_' + c
            tmp.replace([np.inf, -np.inf], np.nan, inplace=True)
            tmp[c] = tmp[c].fillna(tmp.groupby('DATE')[c].transform('median'))
            feather.write_dataframe(tmp, os.path.join(self.savepath, c + '.feather'))

    def run(self):
        self.__daily_calculation()
        # self.cal()


if __name__ == '__main__':
    obj = Big_Order('20250102', '20250701')
    obj.run()
