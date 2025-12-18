"E:\临时用\bin_file\002354_entrust_20241115.csv"
import pandas as pd
import numpy as np
import feather
import os
from sympy.physics.control import Parallel
from tqdm import tqdm
import warnings
from typing import List
from datetime import timedelta, datetime
from collections import defaultdict
from joblib import Parallel, delayed
from tool_tyx.data_path import FactorPath, DataPath

import sys

sys.path.append(r'../')
from tool_tyx.data_path import FactorPath


# 假设已经更新好了daily和五年量,limit_price,后面可以外部调取这个类，进行文件抽取和计算
def find_satisfied_series(sequence: List[float], QGB: int, side: int, vol_ratio_five_years):
    n = len(sequence)
    result = []
    QGP = 0
    QGN = 1
    threshold_amount = QGB * vol_ratio_five_years
    X = 2
    # 遍历所有可能的子序列
    for start in range(n):

        if sequence[start][-1] != side:
            continue

        max_value = sequence[start][-2]
        min_value = sequence[start][-2]
        qgp_count = 0
        in_series_point = []
        in_series_appno = []
        end_max = 0
        qty_sum = 0
        amount_sum = 0
        for end in range(start, n):

            if sequence[end][-1] != side:
                qgp_count += 1
                continue

            if len(in_series_point) == 0:
                max_value = max(max_value, sequence[end][-2])
                min_value = min(min_value, sequence[end][-2])
            else:
                max_value = max(in_series_point)
                min_value = min(in_series_point)

            # 检查当前窗口的最大值和最小值差距
            if abs(sequence[end][-2] - min_value) / min_value > X / 100 or abs(
                    sequence[end][-2] - max_value) / max_value > X / 100:
                qgp_count += 1
            else:
                in_series_point.append(sequence[end][-2])
                in_series_appno.append(sequence[end][1])
                qty_sum += sequence[end][-2]
                amount_sum += sequence[end][-2] * sequence[end][-3]

            # 如果不满足 QGP 条件，停止扩展窗口
            if qgp_count > QGP:
                break

            # 确保起点和终点都满足 QGN 连续条件
            if end - start + 1 >= QGN:
                if (max(in_series_point) - min(in_series_point)) / min(in_series_point) <= X / 100 \
                        and sequence[start][-2] in in_series_point and sequence[end][
                    -2] in in_series_point and (qty_sum > threshold_amount):
                    # 时间要求，暂不考虑
                    # satisfied_time_sig = 0
                    # now_seq = sequence[start:end + 1]
                    # now_seq = [i[-2] for i in now_seq if i[1] in in_series_appno]
                    # for ss in range(1, len(now_seq)):
                    #     if abs(now_seq[ss] - now_seq[ss - 1]) <= self.time_period:
                    #         satisfied_time_sig += 1
                    # if satisfied_time_sig == len(now_seq) - 1:
                    now_seq = sequence[start:end + 1]
                    now_seq = [i for i in now_seq if i[1] in in_series_appno]
                    result.append((sequence[start][1], sequence[end][1], sequence[start:end + 1],
                                   len(in_series_point), qty_sum, now_seq,
                                   amount_sum))

    non_subsequences = []

    if len(result) > 0:
        result.sort(key=lambda x: (x[1], x[1] - x[0]), reverse=True)  # 按序列长度降序排序
        res = [[i[1] for i in x[-2]] for x in result]
        big_order = []  # 存储最终结果
        unique_sets = []
        for i, item in enumerate(res):
            item_set = set(item)
            # 只有当 item 不是任何已选元素的子集时，才保留
            if not any(item_set.issubset(set(checked)) for checked in unique_sets):
                big_order.append(i)

                unique_sets.append(item_set)

        non_subsequences = [list(result[i][2:]) for i in big_order]

    return non_subsequences

# 不用动
def get_this_snapshot_py(order_book, pre_close, max_up, max_down):
    bid_info = {order_book['bid']['bid_price'][i]: sum(order_book['bid']['bid_qty'][i]) for i in
                range(len(order_book['bid']['bid_price']))}
    offer_info = {order_book['offer']['offer_price'][i]: sum(order_book['offer']['offer_qty'][i]) for i in
                  range(len(order_book['offer']['offer_price']))}
    bid_info = {k: bid_info[k] for k in sorted(bid_info, reverse=True)}
    offer_info = {k: offer_info[k] for k in sorted(offer_info)}
    bp = list(bid_info.keys())
    bv = list(bid_info.values())
    for c in range(1, len(bv)):
        bv[c] = bv[c] + bv[c - 1]

    op = list(offer_info.keys())
    ov = list(offer_info.values())
    for c in range(1, len(ov)):
        ov[c] = ov[c] + ov[c - 1]

    all_price = list(set(op + bp))
    all_price.sort()

    if len(bp) == 0:
        sumvolsbuy = [0 for _ in range(len(all_price))]
    else:
        if all_price[-1] == bp[0]:
            sumvolsbuy = [bv[0]]
        else:
            sumvolsbuy = [0]

        for i in range(len(all_price) - 2, -1, -1):
            if all_price[i] in bp:
                sumvolsbuy.append(bv[bp.index(all_price[i])])
            else:
                sumvolsbuy.append(sumvolsbuy[-1])

        sumvolsbuy.reverse()

    if len(op) == 0:
        sumvolssell = [0 for _ in range(len(all_price))]
    else:
        if all_price[0] == op[0]:
            sumvolssell = [ov[0]]
        else:
            sumvolssell = [0]

        for i in range(1, len(all_price)):
            if all_price[i] in op:
                sumvolssell.append(ov[op.index(all_price[i])])
            else:
                sumvolssell.append(sumvolssell[-1])

    auction_vol = [min(sumvolssell[i], sumvolsbuy[i]) for i in range(len(sumvolsbuy))]
    diff_vol = [abs(sumvolssell[i] - sumvolsbuy[i]) for i in range(len(sumvolssell))]
    if len(auction_vol) > 0:
        max_vols = max(auction_vol)
    else:
        max_vols = 0
    # diff_vol = alls['diff_qty'].tolist()
    if max_vols > 0:
        candidates1 = []
        for i in range(len(all_price)):
            if auction_vol[i] == max_vols:
                candidates1.append(all_price[i])

        # 如果只有一个候选，直接选
        if len(candidates1) == 1:
            final_price = candidates1[0]
        else:
            candidates2 = [candidates1[0]]
            min_dff = 1e11
            for idx_min in candidates1:
                if diff_vol[all_price.index(idx_min)] < min_dff:
                    candidates2 = [idx_min]
                    min_dff = diff_vol[all_price.index(idx_min)]
                elif diff_vol[all_price.index(idx_min)] == min_dff:
                    candidates2.append(idx_min)
            if len(candidates2) == 1:
                final_price = candidates2[0]
            else:
                # 如果没有，就先比离昨收价最近
                min_dist = 1e9
                nearest = [candidates1[0]]  # 因为可能有同距离所以初始化为列表
                for price_c in candidates2:
                    dist = abs(price_c - pre_close)
                    if dist < min_dist:
                        min_dist = dist
                        nearest = [price_c]
                    elif dist == min_dist:
                        nearest.append(price_c)
                # 如果距昨收相等，取高价（价格表升序直接取最后一个）
                nearest.sort()
                final_price = nearest[-1]
    else:
        final_price = pre_close

    if final_price in all_price:
        final_price_index = all_price.index(final_price)
        sell_deal, buy_deal = sumvolssell[final_price_index], sumvolsbuy[final_price_index]
        if sell_deal > 0 or buy_deal > 0:
            if buy_deal > sell_deal:
                bid_1_price = final_price
                if len(op) > 0:
                    if np.isclose(final_price, max(op)):
                        offer_1_price = final_price
                    else:
                        higher_asks = [x for x in op if x > final_price]
                        offer_1_price = min(higher_asks) if higher_asks else max_up
                else:
                    offer_1_price = max_up

            elif buy_deal < sell_deal:
                offer_1_price = final_price
                if len(bp) > 0:
                    if np.isclose(final_price, min(bp)):
                        bid_1_price = final_price
                    else:
                        lower_bids = [x for x in bp if x < final_price]
                        bid_1_price = max(lower_bids) if lower_bids else max_down
                else:
                    bid_1_price = max_down
            else:
                offer_1_price = final_price
                bid_1_price = final_price
        else:
            if len(op) > 0:
                if np.isclose(final_price, max(op)):
                    offer_1_price = final_price
                else:
                    higher_asks = [x for x in op if x > final_price]
                    offer_1_price = min(higher_asks) if higher_asks else max_up
            else:
                offer_1_price = max_up

            if len(bp) > 0:
                if np.isclose(final_price, min(bp)):
                    bid_1_price = final_price
                else:
                    lower_bids = [x for x in bp if x < final_price]
                    bid_1_price = max(lower_bids) if lower_bids else max_down
            else:
                bid_1_price = max_down

        auction_qty = min(sell_deal, buy_deal)
        auction_qty_at_auction_price = min(bid_info.get(final_price, 0), offer_info.get(final_price, 0))
        if_is_active_buy = bid_info.get(final_price, 0) > 0
        if_is_active_sell = offer_info.get(final_price, 0) > 0
        return {'offer_1_price': offer_1_price, 'bid_1_price': bid_1_price, '当前撮合价格': final_price,
                '当前撮合成交量': auction_qty,
                '集合竞价价格撮合成交量': auction_qty_at_auction_price,
                '当前时刻是否存在该价格的买单': if_is_active_buy, '当前时刻是否存在该价格的卖单': if_is_active_sell}
    else:
        if len(op) > 0:
            higher_asks = [x for x in op if x > final_price]
            offer_1_price = min(higher_asks) if higher_asks else max_up
        else:
            offer_1_price = max_up

        if len(bp) > 0:
            lower_bids = [x for x in bp if x < final_price]
            bid_1_price = max(lower_bids) if lower_bids else max_down
        else:
            bid_1_price = max_down
        return {'offer_1_price': offer_1_price, 'bid_1_price': bid_1_price, '当前撮合价格': final_price,
                '当前撮合成交量': 0,
                '集合竞价价格撮合成交量': 0, '当前时刻是否存在该价格的买单': False,
                '当前时刻是否存在该价格的卖单': False}


def OrderBook_SZ(order_book, tmp, yesterday_price, max_up, max_down):
    tmp['OrigTime'] = tmp['OrigTime'].astype(np.int64)
    start_time = tmp.groupby('OrigTime')['ApplSeqNum'].count()
    start_time = dict(start_time.T)
    last_price = yesterday_price
    snapshot = {}
    snapshot_py = {}
    # res = []
    for _, r in tmp.iterrows():
        # 循环整个early_order一行一行数据
        if r.ExecType == 'ord':
            if r.Type == '2':
                # 限价单
                if r.Side == 1:
                    if r.Price not in order_book['bid']['bid_price']:
                        order_book['bid']['bid_price'].append(r.Price)
                        order_book['bid']['bid_qty'].append([r.Qty])
                        order_book['bid']['bid_num'].append([r.ApplSeqNum])
                    else:
                        idx = order_book['bid']['bid_price'].index(r.Price)
                        order_book['bid']['bid_qty'][idx].append(r.Qty)
                        order_book['bid']['bid_num'][idx].append(r.ApplSeqNum)

                else:
                    if r.Price not in order_book['offer']['offer_price']:
                        order_book['offer']['offer_price'].append(r.Price)
                        order_book['offer']['offer_qty'].append([r.Qty])
                        order_book['offer']['offer_num'].append([r.ApplSeqNum])
                    else:
                        idx = order_book['offer']['offer_price'].index(r.Price)
                        order_book['offer']['offer_qty'][idx].append(r.Qty)
                        order_book['offer']['offer_num'][idx].append(r.ApplSeqNum)

            elif r.Type == 'U':
                # 如果是本方最优
                if r.Side == 1:
                    # 如果是本方最优买单
                    if len(order_book['bid']['bid_price']) > 0:
                        best_price = max(order_book['bid']['bid_price'])
                        idx = order_book['bid']['bid_price'].index(best_price)
                        order_book['bid']['bid_qty'][idx].append(r.Qty)
                        order_book['bid']['bid_num'][idx].append(r.ApplSeqNum)
                    else:
                        best_price = min(order_book['offer']['offer_price'])
                        order_book['bid']['bid_price'].append(best_price)
                        order_book['bid']['bid_qty'].append([r.Qty])
                        order_book['bid']['bid_num'].append([r.ApplSeqNum])

                else:
                    # 如果是本方最优卖单
                    if len(order_book['offer']['offer_price']) > 0:
                        best_price = min(order_book['offer']['offer_price'])
                        idx = order_book['offer']['offer_price'].index(best_price)
                        order_book['offer']['offer_qty'][idx].append(r.Qty)
                        order_book['offer']['offer_num'][idx].append(r.ApplSeqNum)
                    else:
                        best_price = min(order_book['bid']['bid_price'])
                        order_book['offer']['offer_price'].append(best_price)
                        order_book['offer']['offer_qty'].append([r.Qty])
                        order_book['offer']['offer_num'].append([r.ApplSeqNum])

            else:
                # 如果是市价单
                if r.Side == 1:
                    # 如果是市价买单
                    if len(order_book['offer']['offer_price']) > 0:
                        best_price = min(order_book['offer']['offer_price'])
                        if best_price in order_book['bid']['bid_price']:
                            idx = order_book['bid']['bid_price'].index(best_price)
                            order_book['bid']['bid_qty'][idx].append(r.Qty)
                            order_book['bid']['bid_num'][idx].append(r.ApplSeqNum)
                        else:
                            order_book['bid']['bid_price'].append(best_price)
                            order_book['bid']['bid_qty'].append([r.Qty])
                            order_book['bid']['bid_num'].append([r.ApplSeqNum])
                    else:
                        best_price = max(order_book['bid']['bid_price'])
                        idx = order_book['bid']['bid_price'].index(best_price)
                        order_book['bid']['bid_qty'][idx].append(r.Qty)
                        order_book['bid']['bid_num'][idx].append(r.ApplSeqNum)
                        # do_not_use_app_set.append(r.ApplSeqNum)

                else:
                    # 如果是市价卖单
                    if len(order_book['bid']['bid_price']) > 0:
                        best_price = min(order_book['bid']['bid_price'])

                        if best_price in order_book['offer']['offer_price']:
                            idx = order_book['offer']['offer_price'].index(best_price)
                            order_book['offer']['offer_qty'][idx].append(r.Qty)
                            order_book['offer']['offer_num'][idx].append(r.ApplSeqNum)
                        else:
                            order_book['offer']['offer_price'].append(best_price)
                            order_book['offer']['offer_qty'].append([r.Qty])
                            order_book['offer']['offer_num'].append([r.ApplSeqNum])

                    else:
                        best_price = min(order_book['offer']['offer_price'])
                        idx = order_book['offer']['offer_price'].index(best_price)
                        order_book['offer']['offer_qty'][idx].append(r.Qty)
                        order_book['offer']['offer_num'][idx].append(r.ApplSeqNum)
                        # do_not_use_app_set.append(r.ApplSeqNum)

        else:
            if r.ExecType == '4':
                if r.BidApplSeqNum > 0:
                    #     买方撤单
                    bidno = r.BidApplSeqNum
                    delidx = [x for x in order_book['bid']['bid_num'] if bidno in x][0]
                    idx = order_book['bid']['bid_num'].index(delidx)
                    bidno_idx = order_book['bid']['bid_num'][idx].index(bidno)
                    del order_book['bid']['bid_num'][idx][bidno_idx]
                    del order_book['bid']['bid_qty'][idx][bidno_idx]

                    if len(order_book['bid']['bid_num'][idx]) == 0:
                        del order_book['bid']['bid_num'][idx]
                        del order_book['bid']['bid_qty'][idx]
                        del order_book['bid']['bid_price'][idx]

                else:
                    sellno = r.OfferApplSeqNum
                    #     卖方撤单
                    delidx = [x for x in order_book['offer']['offer_num'] if sellno in x][0]
                    idx = order_book['offer']['offer_num'].index(delidx)
                    sellno_idx = order_book['offer']['offer_num'][idx].index(sellno)
                    del order_book['offer']['offer_num'][idx][sellno_idx]
                    del order_book['offer']['offer_qty'][idx][sellno_idx]

                    if len(order_book['offer']['offer_num'][idx]) == 0:
                        del order_book['offer']['offer_num'][idx]
                        del order_book['offer']['offer_qty'][idx]
                        del order_book['offer']['offer_price'][idx]

            else:
                bidno = r.BidApplSeqNum
                Offerno = r.OfferApplSeqNum
                Qty = r.Qty
                last_price = r.Price

                offer_ = [x for x in order_book['offer']['offer_num'] if Offerno in x][0]
                offer_index = order_book['offer']['offer_num'].index(offer_)
                bid_ = [x for x in order_book['bid']['bid_num'] if bidno in x][0]
                bid_index = order_book['bid']['bid_num'].index(bid_)
                offer_index_at_this_line = order_book['offer']['offer_num'][offer_index].index(Offerno)
                bid_index_at_this_line = order_book['bid']['bid_num'][bid_index].index(bidno)

                order_book['offer']['offer_qty'][offer_index][offer_index_at_this_line] -= min(
                    order_book['offer']['offer_qty'][offer_index][offer_index_at_this_line], Qty)
                order_book['bid']['bid_qty'][bid_index][bid_index_at_this_line] -= min(
                    order_book['bid']['bid_qty'][bid_index][bid_index_at_this_line], Qty)
                if np.isclose(order_book['offer']['offer_qty'][offer_index][offer_index_at_this_line], 0):
                    del order_book['offer']['offer_qty'][offer_index][offer_index_at_this_line]
                    del order_book['offer']['offer_num'][offer_index][offer_index_at_this_line]
                if np.isclose(order_book['bid']['bid_qty'][bid_index][bid_index_at_this_line], 0):
                    del order_book['bid']['bid_qty'][bid_index][bid_index_at_this_line]
                    del order_book['bid']['bid_num'][bid_index][bid_index_at_this_line]

                if len(order_book['bid']['bid_num'][bid_index]) == 0:
                    del order_book['bid']['bid_num'][bid_index]
                    del order_book['bid']['bid_qty'][bid_index]
                    del order_book['bid']['bid_price'][bid_index]

                if len(order_book['offer']['offer_num'][offer_index]) == 0:
                    del order_book['offer']['offer_num'][offer_index]
                    del order_book['offer']['offer_qty'][offer_index]
                    del order_book['offer']['offer_price'][offer_index]

        start_time[r.OrigTime] -= 1

        if r.time >= 91500000:
            snapshot[r.ApplSeqNum] = get_this_snapshot_py(order_book, yesterday_price, max_up, max_down)

    auction_price = pd.DataFrame(snapshot).T.reset_index(drop=False)
    return auction_price.rename(columns={'index': 'ApplSeqNum'})


def get_sz_orderbook(tmp_order, tmp_trade, pre_close, max_up, max_down):
    order_book = {'bid': {'bid_price': [], 'bid_qty': [], 'bid_num': []},
                  'offer': {'offer_price': [], 'offer_qty': [], 'offer_num': []}}
    ticker = tmp_trade.SecurityID.values[0] + '.SZ'

    tmp_order_ = tmp_order[['ApplSeqNum', 'Price', 'OrderQty', 'Side', 'OrderType', 'OrigTime']]
    tmp_order_.columns = ['ApplSeqNum', 'Price', 'Qty', 'Side', 'Type', 'OrigTime']
    tmp_order_['ExecType'] = 'ord'
    tmp_order_['BidApplSeqNum'] = 0
    tmp_order_['OfferApplSeqNum'] = 0

    tmp_trade_ = tmp_trade[
        ['ApplSeqNum', 'Price', 'TradeQty', 'ExecType', 'BidApplSeqNum', 'OfferApplSeqNum', 'OrigTime']]
    tmp_trade_.columns = ['ApplSeqNum', 'Price', 'Qty', 'ExecType', 'BidApplSeqNum', 'OfferApplSeqNum', 'OrigTime']
    tmp_trade_['Type'] = 'T'
    tmp_trade_['Side'] = 'T'
    temp = pd.concat([tmp_order_, tmp_trade_]).sort_values(by=['ApplSeqNum']).reset_index(drop=True)
    temp['time'] = temp['OrigTime'].astype(str).str[8:].astype(np.int64)
    temp = temp[temp['time'] < 93000000]
    orderbook = OrderBook_SZ(order_book, temp, yesterday_price=pre_close, max_up=max_up, max_down=max_down)

    orderbook['上一笔撮合价格'] = orderbook['当前撮合价格'].shift(1).fillna(pre_close)
    orderbook['上一时刻撮合成交量'] = orderbook['当前撮合成交量'].shift(1).fillna(0)
    orderbook['上一时刻是否存在集合竞价价格的卖单'] = orderbook['当前时刻是否存在该价格的卖单'].shift(1).bfill()
    orderbook['上一时刻是否存在集合竞价价格的买单'] = orderbook['当前时刻是否存在该价格的买单'].shift(1).bfill()

    use_Data = temp[
        ['ApplSeqNum', 'Price', 'Qty', 'Side', 'Type', 'ExecType', 'BidApplSeqNum', 'OfferApplSeqNum', 'time']]
    use_Data = use_Data.rename(columns={'time': 'tradetime'})
    use_Data = pd.merge(use_Data, orderbook, how='left', on=['ApplSeqNum'])

    use_Data = use_Data.rename(columns={'Qty': 'TradeQty', 'ExecType': 'type'})
    use_Data.loc[use_Data['type'] == 'ord', 'type'] = 'A'
    use_Data = use_Data[
        ['ApplSeqNum', 'tradetime', 'Price', 'TradeQty', 'Side', 'type', '当前撮合价格', '上一笔撮合价格',
         '当前撮合成交量',
         '上一时刻是否存在集合竞价价格的卖单', '上一时刻是否存在集合竞价价格的买单', '集合竞价价格撮合成交量',
         '上一时刻撮合成交量',
         ]]
    use_Data = use_Data[use_Data['type'] != 'F'].sort_values(by=['ApplSeqNum']).reset_index(drop=True)

    return use_Data


def run_one_tic(info):
    warnings.filterwarnings(action='ignore')
    tic, date, max_up, max_down, previous_close, past_5_vol, daily_save_path = info[0], info[1], info[2], info[3], info[
        4], info[5], info[6]
    vol_ratio_five_years = 0.0005
    if date[:4] == '2025':
        datapath_this = DataPath.feather_2025
    elif date[:4] == '2024':
        datapath_this = DataPath.feather_2024
    elif date[:4] == '2023':
        datapath_this = DataPath.feather_2023
    elif date[:4] == '2022':
        datapath_this = DataPath.feather_2022
    else:
        return pd.DataFrame()

    try:
        if os.path.exists(rf'{datapath_this}\{date}\hq_trade_spot\{tic[:6]}.feather') and os.path.exists(
                rf'{datapath_this}\{date}\hq_order_spot\{tic[:6]}.feather'):
            trade = feather.read_dataframe(
                rf'{datapath_this}\{date}\hq_trade_spot\{tic[:6]}.feather')
            order = feather.read_dataframe(
                rf'{datapath_this}\{date}\hq_order_spot\{tic[:6]}.feather')

            # 如果uo满足你的需求就不用动 get this snapshot py
            uo = get_sz_orderbook(order, trade, previous_close, max_up=max_up, max_down=max_down)
            uo = uo[(uo['tradetime'] >= 92000000)&(uo['tradetime'] < 92459950)]
            uo['sec'] = uo['tradetime']//1000
            split_by_sec = uo.groupby('sec').agg(
                {'当前撮合成交量': ['first', 'last'], '当前撮合价格': ['first', 'last', 'max', 'min']}).reset_index(
                drop=False)
            split_by_sec.columns = ['sec', 'vol_start', 'vol_end', 'open', 'close', 'high', 'low']
            split_by_sec['volume_diff'] = split_by_sec['vol_end'] - split_by_sec['vol_start']
            split_by_sec['volatility'] = 100 * (split_by_sec['high'] / split_by_sec['low'] - 1)

            xdif, retdiff = np.diff(split_by_sec['volume_diff']), np.diff(split_by_sec['volatility'])
            if xdif.size == 0:
                return
            threshold = xdif.mean() + xdif.std()
            split_by_sec = split_by_sec.iloc[1:, :]
            split_by_sec['retdiff'], split_by_sec['mvdif'] = retdiff, xdif
            split_by_sec = split_by_sec.reset_index(drop=True)
            g, res = split_by_sec[split_by_sec['mvdif'] > threshold].index, []
            for i in g:
                if i < split_by_sec.shape[0] - 5:
                    res.append(split_by_sec['retdiff'].iloc[i:i + 5].std())
            # 先将不存在日耀波动率的股票的值设0
            # 月耀眼波动率如果在这一天不存在的话,就证明当天的股价波动小
            # 最后要求均值的话如果为全零填充也将会拉低月均耀眼波动率,这与实际相符
            a = sum(res) / len(res) if res else 0
            res = pd.DataFrame({'TICKER': tic, 'DATE': date, 'MSF_AUCTION': a}, index=[0])
            return res
    except:
        return pd.DataFrame()


def clean_sz_data(datels, jobuse, filename):
    warnings.filterwarnings(action='ignore')
    low_frequency_ip = '192.168.1.12'
    daily_save_path = os.path.join(rf'\\{low_frequency_ip}\ssd\output_data', 'tyx', filename, 'concat_daily')
    os.makedirs(daily_save_path, exist_ok=True)

    datapath = FactorPath.data_storage
    daily = feather.read_dataframe(os.path.join(datapath, 'daily.feather'),
                                   columns=['TICKER', 'DATE', 'max_up', 'max_down'])
    daily = daily[daily['DATE'].isin(datels)]
    daily = daily[daily['TICKER'].str[-2:] == 'SZ'].reset_index(drop=True)

    previous_close_all = feather.read_dataframe(os.path.join(datapath, 'adj_pre_close.feather'))
    previous_close_all = previous_close_all[['TICKER', 'DATE', 'pre_close']]
    previous_close_all.columns = ['TICKER', 'DATE', 'close']
    previous_close_all = previous_close_all[previous_close_all['DATE'].isin(datels)]
    previous_close_all = previous_close_all[previous_close_all['TICKER'].str[-2:] == 'SZ'].reset_index(drop=True)

    past_5_max = feather.read_dataframe(os.path.join(datapath, 'max_volume_in_past_1200_days_pre_adj.feather'))
    past_5_max = past_5_max[past_5_max['DATE'].isin(datels)]
    past_5_max = past_5_max[past_5_max['TICKER'].str[-2:] == 'SZ'].reset_index(drop=True)

    past_5_max = past_5_max.rename(
        columns={np.setdiff1d(past_5_max.columns, ['TICKER', 'DATE'])[0]: 'vol'})
    vol_ratio_five_years = 0.0005

    for date in datels:
        int_ticker1 = past_5_max[past_5_max['DATE'] == date]['TICKER'].unique()
        int_ticker2 = daily[daily['DATE'] == date]['TICKER'].unique()
        int_ticker3 = previous_close_all[previous_close_all['DATE'] == date]['TICKER'].unique()
        int_tic = np.intersect1d(int_ticker1, int_ticker2)
        int_tic = np.intersect1d(int_tic, int_ticker3)
        int_tic = list(int_tic)

        use_data = daily[daily['DATE'] == date]
        use_data = use_data[use_data['TICKER'].isin(int_tic)]
        use_data = use_data[['TICKER', 'DATE', 'max_up', 'max_down']]
        use_data = pd.merge(use_data, previous_close_all, how='left', on=['TICKER', 'DATE'])
        use_data = pd.merge(use_data, past_5_max, how='left', on=['TICKER', 'DATE'])
        use_data = use_data[['TICKER', 'DATE', 'max_up', 'max_down', 'close', 'vol']]
        use_data['path'] = daily_save_path
        use_data = use_data.values.tolist()
        if jobuse > 1:
            res = Parallel(n_jobs=jobuse)(delayed(run_one_tic)(info) for info in tqdm(use_data, desc=date))
        else:
            res = []
            for info in tqdm(use_data, desc=date):
                res.append(run_one_tic(info))
        res = pd.concat(res).reset_index(drop=True)
        feather.write_dataframe(res, rf'{daily_save_path}\{date}.feather')


class big_order_select(object):
    def __init__(self, start=None, end=None, bin_path=None, exe_path=None, daily_path=None, past_5_max_path=None,
                 basic_savepath=None, output_savepath=None,
                 up_threshold=None, limit_price_path=None, vol_ratio_five_years=None,
                 price_upper_threshold_percent=None,
                 amount_threshold=None, jobuse=None, cluster=None):
        self.start = start or '20220101'
        self.end = end or '20241231'
        # 日k，五年量，复权因子
        self.datapath = daily_path or os.path.join(r'\\192.168.1.210\Data_Storage2', 'daily.feather')
        # 数据，输出路径
        self.jobuse = jobuse or 24
        self.cluster = cluster or {'high_frequency_ip': '192.168.1.7', 'low_frequency_ip': '192.168.1.12'}
        self.low_frequency_ip = '192.168.1.12'

    def load_basic_data(self):
        warnings.filterwarnings(action='ignore')
        daily = feather.read_dataframe(self.datapath)
        datels = daily[['DATE']]
        datels = daily['DATE'].unique().tolist()
        datels = [i for i in datels if i <= self.end and i >= self.start]
        self.datels = datels

    def get_all_results_sz(self):
        print('start calculating data')
        low_frequency_ip = self.low_frequency_ip
        filename = 'MSF_AUCTION_SZ_ALL'
        ls = os.path.join(rf'\\{low_frequency_ip}\ssd\output_data', 'tyx', filename, 'concat_daily')
        os.makedirs(ls, exist_ok=True)
        already = [i[:8] for i in os.listdir(ls)]
        self.datels = [i for i in self.datels if i not in already]

        clean_sz_data(self.datels, jobuse=self.jobuse, filename=filename)

    def run(self):
        self.load_basic_data()
        self.get_all_results_sz()


if __name__ == "__main__":
    obj = big_order_select(start='20251112', end='20251112', jobuse=1, basic_savepath=DataPath.feather_2025)
    obj.run()
