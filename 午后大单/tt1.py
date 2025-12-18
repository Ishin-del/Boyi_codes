import warnings
from datetime import datetime
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

        def get_code(date,code):
            warnings.filterwarnings('ignore')
            dt = feather.read_dataframe(os.path.join(self.df_path, fr'{date}\hq_trade_spot\{code}'))
            dt.drop(columns=['OrigTime','index', 'tradedate', 'ChannelNo', 'MDStreamID', 'SecurityIDSource'], inplace=True)
            dt.sort_values('ApplSeqNum', inplace=True)
            try:
                open_price=dt[dt['tradetime']==int(f'{date}092500000')]['Price'].iloc[0]
            except:
                return
            dt = dt[dt['tradetime'] <= int(f'{date}130000000') + 6000].reset_index(drop=True)
            # 合并order
            order_dt = feather.read_dataframe(os.path.join(self.df_path, fr'{date}\hq_order_spot\{code}'))
            order_dt.drop(columns=['index', 'tradedate', 'OrigTime', 'ChannelNo', 'MDStreamID', 'SecurityID','OrderType'],inplace=True)
            price_map = order_dt.set_index('ApplSeqNum')['Price']
            zero_mask = dt['Price'] == 0
            seq_nums = dt['BidApplSeqNum'] + dt['OfferApplSeqNum']
            dt.loc[zero_mask, 'Price'] = seq_nums[zero_mask].map(price_map)
            # ------------------------------------
            # 计算1s内的委托量，并保证 委托的价格大于等于最近一笔的成交价（买方）
            dt['最近一笔成交价'] = np.where(dt['ExecType'] == 'F', dt['Price'], np.nan)  # 如果成交用当前价格，否则填nan值，然后用前面数据填充
            dt['最近一笔成交价'] = dt['最近一笔成交价'].ffill()
            tar_df = dt[(dt['tradetime'] >= int(f'{date}130000000'))&(dt['tradetime'] <= int(f'{date}130001000')) & (dt['BidApplSeqNum'] > dt['OfferApplSeqNum']) &
                        (dt['Price'] >= dt['最近一笔成交价'])&(dt['ExecType'] == 'F')]
            tar_df_num = tar_df.groupby('BidApplSeqNum').agg({'ApplSeqNum': 'last','tradetime':'last'})  # 一笔订单分几波成交，按最后成交结束对应的时间来定格
            tar_df = tar_df.groupby('BidApplSeqNum').agg({'TradeQty': 'sum'}).rename(columns={'TradeQty': 'TradeQty_sum'})
            tar_df = tar_df.merge(tar_df_num, right_index=True, left_index=True, how='outer').reset_index()
            code = code.replace('.feather', '.SZ')
            vols_threshold=self.threshold_df[(self.threshold_df['DATE']==date)&(self.threshold_df['TICKER']==code)]
            if vols_threshold.empty:
                return
            vols_threshold=vols_threshold['vols_threshold'].iloc[0]*0.4
            tar_df = tar_df[tar_df['TradeQty_sum'] >= vols_threshold]  # tar_df选出大于等于阈值的行
            if tar_df.empty:
                return

            res = []
            for _, row in tar_df.iterrows():
                big_order_time = row['tradetime']
                curr_df=dt[(dt['BidApplSeqNum']==row['BidApplSeqNum'])&(dt['tradetime']<=big_order_time+5000)]
                curr_df.sort_values('ApplSeqNum',inplace=True)
                # curr_df是序列，序列是取该大单单号，观察范围在5s内
                curr_trade_df=curr_df[curr_df['ExecType']=='F'] #序列中 成交的数据 （主动+被动）
                curr_trade_act_df=curr_trade_df[curr_trade_df['BidApplSeqNum']>curr_trade_df['OfferApplSeqNum']]
                curr_order_df=order_dt[order_dt['ApplSeqNum'] == row['BidApplSeqNum']]
                #========================================================================
                first_price=curr_trade_act_df['Price'].iloc[0] # 序列第一笔主动成交价格（基准价格）
                last_price=curr_trade_act_df['Price'].iloc[-1] # 序列最后一笔主动成交价格
                dt.sort_values('ApplSeqNum',inplace=True)
                trade_price_before=dt[(dt['ExecType']=='F')&(dt['tradetime']<=big_order_time-10)]['Price'].iloc[-1] #股票在序列发出后一个时刻（10ms）连续竞价的成交价
                trade_price_after=dt[(dt['ExecType']=='F')&(dt['tradetime']<=curr_df['tradetime'].iloc[-1]+10)]['Price'].iloc[0] #股票在序列发出后一个时刻（10ms）连续竞价的成交价
                price_ret=last_price/first_price-1 # 序列最后一笔主动成交价格相对于基准价格涨跌幅
                trade_ret_before_open=trade_price_before/open_price-1 # 序列发出前连续竞价成交价格涨幅
                trade_ret_after_open=trade_price_after/open_price-1 # 序列发出前连续竞价成交价格涨幅
                before_after_diff=trade_ret_after_open-trade_ret_before_open
                #委托价格 => order里面找
                firstOrderPrice=curr_order_df['Price'].iloc[0]
                orderPrice = firstOrderPrice
                firstOrderToBefore=firstOrderPrice/trade_price_before-1
                firstOrderToTrade=firstOrderPrice/first_price-1
                OrderToOpen=firstOrderPrice/open_price-1
                firstToOpen=first_price/open_price-1
                # --------------------------------------
                orderQty=curr_order_df['OrderQty'].iloc[0] #序列总委托量(手),到order里找
                orderQtyToThreshold=orderQty/vols_threshold
                tradeQty=curr_trade_df['TradeQty'].sum() # 委托序列在连续竞价成交量(手)
                tradeToOrder=tradeQty/orderQty
                cancelQty=curr_df[curr_df['ExecType']=='4']['TradeQty'].sum() #委托序列在连续竞价撤单量(手)
                cancelToOrder = cancelQty / orderQty
                orderAmt=firstOrderPrice*orderQty
                act_df=curr_trade_df[curr_trade_df['BidApplSeqNum'] > curr_trade_df['OfferApplSeqNum']]
                act_volume = act_df['TradeQty'].sum()  # 主动成交量
                pass_df=curr_trade_df[curr_trade_df['BidApplSeqNum'] < curr_trade_df['OfferApplSeqNum']]
                pass_volume=pass_df['TradeQty'].sum() #被动成交量
                actToTrade=act_volume/tradeQty
                passToTrade=pass_volume/tradeQty
                actToOrder=act_volume/orderQty
                passToOrder=pass_volume/orderQty
                actAmt=(act_df['TradeQty']*act_df['Price']).sum()
                passAmt=(pass_df['TradeQty']*pass_df['Price']).sum()
                actToThreshold=act_volume/vols_threshold
                exist_act= act_volume!=0
                if not exist_act:
                    end_df=curr_df[['Price','tradetime']].tail(1)
                else:
                    end_df=curr_df[curr_df['ExecType']=='F'][['Price','tradetime']].tail(1)
                the_time=curr_df[curr_df['Price']>=end_df['Price'].iloc[0]][['tradetime']].tail(1)['tradetime'].iloc[0]
                if end_df['tradetime'].iloc[0]<the_time:
                    time_diff=0
                else:
                    time_diff=datetime.strptime(str(end_df['tradetime'].iloc[0])[:14],'%Y%m%d%H%M%S')-datetime.strptime(str(the_time)[:14],'%Y%m%d%H%M%S')
                    time_diff=time_diff.seconds/60 # 序列结束后这么些分钟创最高价
                # -------------------------------------------------
                # tmp_order=order_dt[order_dt['ApplSeqNum'].isin(curr_trade_df['BidApplSeqNum'])]
                # price_list=tmp_order['Price'].to_list()
                # orderqty_list=tmp_order['OrderQty'].to_list()
                # app_list=tmp_order['ApplSeqNum'].to_list()

                res.append([date,code,big_order_time,first_price,last_price,trade_price_before,trade_price_after,price_ret,
                            trade_ret_before_open, trade_ret_after_open, before_after_diff, firstOrderToBefore,
                            firstOrderToTrade,orderPrice,OrderToOpen, firstToOpen,orderQtyToThreshold,orderQty,tradeQty,tradeToOrder,
                            cancelToOrder,cancelQty,orderAmt,act_volume,pass_volume,actToTrade,passToTrade,actToOrder,
                            passToOrder,actAmt,passAmt,actToThreshold,exist_act,time_diff])
                # ,price_list,orderqty_list,app_list
            # print(res)
            return res

        def get_daily(date):
            # df=get_code('20250102','002285.feather')
            # df=pd.DataFrame(df)
            # print(df)
            # # tar_codes=['002995.feather','000001.feather','000004.feather']
            tar_codes=os.listdir(os.path.join(self.df_path,fr'{date}\hq_trade_spot'))
            total_res=[]
            for code in tqdm(tar_codes):
                total_res.append(get_code(date,code))
            total_res = [x for x in total_res if x != None]
            df = pd.DataFrame(list(chain.from_iterable(total_res)))
            df.columns=['DATE','TICKER','序列开始时间','序列第一笔主动成交价格(基准价格)','序列最后一笔主动成交价格','序列发出前连续竞价成交价格',
                '序列最后一笔委托发出后连续竞价成交价格','序列最后一笔主动成交价格相对于基准价格涨跌幅','序列开始前成交涨幅','序列结束后成交涨幅',
                '序列开始前后市场涨幅','序列第一笔委托价格相对于序列发出前连续竞价成交价格涨幅','序列第一笔委托价格相对于基准价格涨幅',
                '委托价格','委托价格相对开盘涨幅','基准价格相对开盘涨幅','序列委托量占比阈值','序列总委托量(手)','委托序列在连续竞价成交量(手)',
                '委托序列在连续竞价成交比例','委托序列在连续竞价撤单比例','委托序列在连续竞价撤单量(手)','序列委托金额','序列主动买入成交量(手)',
                '序列被动买入成交量(手)','序列主动买入成交量占总成交量比例','序列被动买入成交量占总成交量比例','序列主动买入成交量占总委托量比例',
                '序列被动买入成交量占总委托量比例','序列主动买入成交额(万)','序列被动买入成交额(万)','委托序列主动买入量占阈值比例','是否存在主动买入',
                '序列结束后创过去多少分钟的最高价']
            # df=df.explode(['序列分笔委托价格','序列分笔委托量(手)','序列分笔单号'])
            # df.sort_values('序列分笔单号',inplace=True)
            print(df)
            feather.write_dataframe(df,r'C:\Users\admin\Desktop\20250102.feather')
            df.to_csv(r'C:\Users\admin\Desktop\20250102.csv',encoding='gbk',index=False)
            # # return df

        df=get_daily('20250102')
        # print(df)
        # df.to_csv(r'C:\Users\admin\Desktop\tt.csv')
        # df = Parallel(n_jobs=10)(delayed(get_daily)(date) for date in tqdm(date_list))
        # df = pd.concat(df).reset_index(drop=True)
        # alls = pd.concat([df_old, df])
        # alls =alls.drop_duplicates(subset=['TICKER','DATE'],keep='first').reset_index(drop=True)
        # feather.write_dataframe(alls, os.path.join(self.tmp_path, 'high_freq_skew.feather'))

    def run(self):
        self.__daily_calculation()
        # self.cal()


if __name__ == '__main__':
    obj = Big_Order('20250102', '20250701')
    obj.run()
