import pandas as pd
import numpy as np

def reconstruct_sh_tick(order_sh, trade_sh):
    '''
    将服务器上读取返回的分开的上海的逐笔委托和逐笔成交数据进行整合成24年之后交易所返回的完整的一个文件
    返回的df与原始数据仅仅相差TickBsFlag为OCALL，TRADE，CCALL，CLOSE，ENDTR这5条提示交易时段开始结束的记录（非交易记录）
    '''
    # 合并上海的数据回退为原来的结构
    recovered_order = order_sh.rename(columns={
        'TransactTime': 'TickTime',
        'Balance': 'Qty',
        'OrderBSFlag': 'TickBSFlag',
        'OrdType': 'Type',
        'OrderIndex': 'TickIndex'
    }).copy()
    recovered_order['BuyOrderNo'] = 0
    recovered_order['SellOrderNo'] = 0
    recovered_order.loc[recovered_order['TickBSFlag'] == 'B', 'BuyOrderNo'] = recovered_order['OrderNo']
    recovered_order.loc[recovered_order['TickBSFlag'] == 'S', 'SellOrderNo'] = recovered_order['OrderNo']

    recovered_trade = trade_sh.rename(columns={
        'TradeTime': 'TickTime',
        'TradePrice': 'Price',
        'TradeQty': 'Qty',
        'TradeAmount': 'TradeMoney',
        'BuyNo': 'BuyOrderNo',
        'SellNo': 'SellOrderNo',
        'TradeIndex': 'TickIndex',
        'TradeBSFlag': 'TickBSFlag',
    }).copy()
    recovered_trade['Type'] = 'T'
    recovered_trade['TradeMoney'] = recovered_trade['TradeMoney'].astype('float')
    recovered_trade['OrderNo'] = np.nan  

    stock_tick_restored = pd.concat([recovered_order, recovered_trade], ignore_index=True)
    if 'BizIndex' in stock_tick_restored.columns:
        stock_tick_restored['TickIndex'] = stock_tick_restored['BizIndex']
    stock_tick_restored.reset_index(drop=True)
    stock_tick_restored.sort_values(by='TickIndex', inplace=True)
    return stock_tick_restored


def reconstruct_sh_orderl2(raw_sh_tickdata):
    '''
    依据深圳的逐笔委托数据情况，使用上海逐笔数据还原上海的逐笔委托订单详细情况
    '''
    new_order_sh = {} ## 存储所有的委托订单信息
    newBuyTbeforeA_status_dict = {}
    newSellTbeforeA_status_dict = {}
    maxBuyOrderNo, maxSellOrderNo = 0, 0
    for _, row in raw_sh_tickdata.iterrows():
        TickTime = row['TickTime']
        buyOrderNo = row['BuyOrderNo']
        sellOrderNo = row['SellOrderNo']
        tradeType = row['Type']
        OrderQty = row['Qty']
        tradePrice = row['Price']
        # 新增状态追踪，确保T之前不会有A的类型
        newBuyTbeforeA_status = newBuyTbeforeA_status_dict.get(buyOrderNo, True)
        newSellTbeforeA_status = newSellTbeforeA_status_dict.get(sellOrderNo, True)
        new_order_row = {}
        if buyOrderNo != 0 and sellOrderNo == 0 and tradeType == 'A' and buyOrderNo > maxBuyOrderNo:
            # print(f'新增买单委托{buyOrderNo}, 订单类型为{tradeType}')
            new_order_row['OrigTime'] = TickTime
            new_order_row['ApplSeqNum'] = buyOrderNo
            new_order_row['Price'] = tradePrice
            new_order_row['OrderQty'] = OrderQty
            new_order_row['Side'] = 1
            new_order_row['OrderType'] = 'A'
            new_order_sh[buyOrderNo] = new_order_row
            maxBuyOrderNo = buyOrderNo
            newBuyTbeforeA_status_dict[buyOrderNo] = False
        elif buyOrderNo == 0 and sellOrderNo != 0 and tradeType == 'A' and sellOrderNo > maxSellOrderNo:
            # print(f'新增卖单委托{sellOrderNo}, 订单类型为{tradeType}')
            new_order_row['OrigTime'] = TickTime
            new_order_row['ApplSeqNum'] = sellOrderNo
            new_order_row['Price'] = tradePrice
            new_order_row['OrderQty'] = OrderQty
            new_order_row['Side'] = 2
            new_order_row['OrderType'] = 'A'
            new_order_sh[sellOrderNo] = new_order_row
            maxSellOrderNo = sellOrderNo
            newSellTbeforeA_status_dict[sellOrderNo] = False
        elif buyOrderNo != 0 and sellOrderNo == 0 and tradeType == 'A' and buyOrderNo <= maxBuyOrderNo:
            # print(f'前已挂单成交订单继续增加委买{buyOrderNo}, 订单类型为{tradeType}')
            if buyOrderNo in new_order_sh:
                # 对原来的数据进行更新，主要进行挂单量和挂单类型的更新，不改变初始的挂单时间，对于一个订单通常只记录挂单时间，撤单的时间额外处理
                new_order_sh[buyOrderNo]['OrderQty'] += OrderQty
                new_order_sh[buyOrderNo]['OrderType'] = 'TA'
                newBuyTbeforeA_status_dict[buyOrderNo] = False # 更新状态，不再统计后续的T订单的量进委托量
        elif buyOrderNo == 0 and sellOrderNo != 0 and tradeType == 'A' and sellOrderNo <= maxSellOrderNo:
            # print(f'前已挂单成交订单继续增加委卖{sellOrderNo}, 订单类型为{tradeType}')
            if sellOrderNo in new_order_sh:
                # 对原来的数据进行更新，主要进行挂单量和挂单类型的更新，不改变初始的挂单时间，对于一个订单通常只记录挂单时间，撤单的时间额外处理
                new_order_sh[sellOrderNo]['OrderQty'] += OrderQty
                new_order_sh[sellOrderNo]['OrderType'] = 'TA'
                newSellTbeforeA_status_dict[sellOrderNo] = False # 更新状态，不再统计后续的T订单的量进委托量            
        elif tradeType == 'T':
            # 处理即时成交订单的委托量还原
            if buyOrderNo > maxBuyOrderNo:
                # 新增即时成交买单
                # print(f'新增即时成交委托买单挂单{buyOrderNo}, 订单类型为{tradeType}')
                new_order_row['OrigTime'] = TickTime
                new_order_row['ApplSeqNum'] = buyOrderNo
                new_order_row['Price'] = tradePrice
                new_order_row['OrderQty'] = OrderQty
                new_order_row['Side'] = 1
                new_order_row['OrderType'] = 'T'
                maxBuyOrderNo = buyOrderNo
                new_order_sh[buyOrderNo] = new_order_row
            elif buyOrderNo <= maxBuyOrderNo and newBuyTbeforeA_status:
                # print(f'更新前增即时成交委托买单挂单{buyOrderNo}, 订单类型为{tradeType}')
                if buyOrderNo in new_order_sh:
                    new_order_sh[buyOrderNo]['OrderQty'] += OrderQty
                    new_order_sh[buyOrderNo]['OrderType'] = 'T'
                    
            if sellOrderNo > maxSellOrderNo:
                # 新增即时成交卖单
                # print(f'新增即时成交委托卖单挂单{sellOrderNo}, 订单类型为{tradeType}')
                new_order_row['OrigTime'] = TickTime
                new_order_row['ApplSeqNum'] = sellOrderNo
                new_order_row['Price'] = tradePrice
                new_order_row['OrderQty'] = OrderQty
                new_order_row['Side'] = 2
                new_order_row['OrderType'] = 'T'
                maxSellOrderNo = sellOrderNo
                new_order_sh[sellOrderNo] = new_order_row
            elif sellOrderNo <= maxSellOrderNo and newSellTbeforeA_status:
                # print(f'更新前增即时成交委托卖单挂单{sellOrderNo}, 订单类型为{tradeType}')
                if sellOrderNo in new_order_sh:
                    new_order_sh[sellOrderNo]['OrderQty'] += OrderQty
                    new_order_sh[sellOrderNo]['OrderType'] = 'T'
        elif tradeType == 'D':
            # 处理撤单记录
            if buyOrderNo > 0:
                # 此时为买单撤单记录
                # print(f'新增买单撤单{sellOrderNo}, 订单类型为{tradeType}')
                new_order_row['OrigTime'] = TickTime
                new_order_row['ApplSeqNum'] = buyOrderNo
                new_order_row['Price'] = tradePrice
                new_order_row['OrderQty'] = OrderQty
                new_order_row['Side'] = 1
                new_order_row['OrderType'] = 'D'
                new_order_sh[(buyOrderNo, 'D')] = new_order_row
            if sellOrderNo > 0:
                # 此时为卖单撤单记录
                # print(f'新增卖单撤单{sellOrderNo}, 订单类型为{tradeType}')
                new_order_row['OrigTime'] = TickTime
                new_order_row['ApplSeqNum'] = sellOrderNo
                new_order_row['Price'] = tradePrice
                new_order_row['OrderQty'] = OrderQty
                new_order_row['Side'] = 2
                new_order_row['OrderType'] = 'D'
                new_order_sh[(sellOrderNo, 'D')] = new_order_row
                
    new_order_sh = pd.DataFrame(new_order_sh.values())            
    new_order_sh['tradedate'] = raw_sh_tickdata['TickTime'].astype(str).apply(lambda x: x[:8]).unique()[0]
    new_order_sh['ChannelNo'] = raw_sh_tickdata['ChannelNo'].unique()[0]
    new_order_sh['SecurityID'] = raw_sh_tickdata['SecurityID'].unique()[0]
    new_order_sh['TransactTime'] = new_order_sh['OrigTime']
    desired_order = ['tradedate', 'OrigTime', 'ChannelNo', 'ApplSeqNum', 'SecurityID', 
                    'Price', 'OrderQty', 'TransactTime', 'Side', 'OrderType']
    new_order_sh = new_order_sh[desired_order]
    new_order_sh.sort_values(by=['ApplSeqNum'], inplace=True)
    new_order_sh.reset_index(drop=True)
    return new_order_sh


def rename_sh_trade(trade_sh):
    trade_sh['tradedate'] = trade_sh['TradeTime'].astype(str).apply(lambda x: x[:8]).unique()[0]
    trade_sh.rename(columns={
        'BuyNo': 'BidApplSeqNum',
        'SellNo': 'OfferApplSeqNum',
        'TradePrice': 'Price',
        'TradeTime': 'tradetime',
        'TradeIndex': 'ApplSeqNum'
    }, inplace=True)
    trade_sh['ExecType'] = 'F'
    # 上海逐笔成交数据时间精度在24年下降了10ms级别，乘10进行统一到17位数的时间精度
    if trade_sh['tradetime'].apply(lambda x: len(str(x)) == 16).all():
        trade_sh['OrigTime'] = trade_sh['tradetime'] * 10
    else:
        trade_sh['OrigTime'] = trade_sh['tradetime']
    trade_sh.reset_index(drop=True)
    return trade_sh