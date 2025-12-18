import feather


order=feather.read_dataframe(r'X:\data_feather\20250102\hq_order_spot\000001.feather')
order['Side'] = order['Side'].astype(int)
order['ApplSeqNum'] = order['ApplSeqNum'].astype(int)
# trade
trade=feather.read_dataframe(r'X:\data_feather\20250102\hq_trade_spot\000001.feather')
trade.drop(columns=['index', 'tradedate', 'OrigTime', 'ChannelNo', 'MDStreamID','SecurityIDSource'],inplace=True)
trade['BidApplSeqNum'] = trade['BidApplSeqNum'].astype(int)
trade['OfferApplSeqNum'] = trade['OfferApplSeqNum'].astype(int)
print('sdf')

"""
集合竞价期间没有成交的单子会变成废单，不会体现在撤单中
而是委托订单可能分笔在多个时间段中成交，所以截时间段中操作也不行
# trade里面只记录有效单（这个概念主要是去区分废单），无论集合竞价的废单or连续竞价的废单，都不会出现在trade里，只会出现在order里

碰到找委托单的直接从逐笔委托数据里面找
================================
order => 单方行为
trade => 双方行为
在trade里：
bid>offer, 主动买且被动卖
offer>bid, 被动买且主动卖
"""
tt = trade.groupby('BidApplSeqNum').agg({'TradeQty':'sum'}).reset_index()
tt.columns=['ApplSeqNum', 'TradeQty']
tt1 = order[order['Side'] == 1]
ss = tt.merge(tt1[['ApplSeqNum', 'OrderQty']], on='ApplSeqNum', how='outer')
ss[ss['TradeQty'].isna()]