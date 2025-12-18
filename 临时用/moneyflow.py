import feather
import pandas as pd
import numpy as np
import os
import time


def calculate_effective_orders_corrected(order_data):
    """
    计算有效的委托订单
    先按订单号计算净委托量，再分配到分钟
    """
    # 复制数据避免修改原数据
    order_df = order_data.copy()

    # 创建min列（如果不存在的话）
    if 'min' not in order_df.columns:
        order_df['min'] = (order_df['TickTime'].astype(str).str[-9:-5]).astype('int32')

    # 为买卖方向创建统一的订单号列
    order_df['OrderNo'] = np.where(order_df['Side'] == 1,
                                   order_df['BuyOrderNo'],
                                   order_df['SellOrderNo'])

    # 为A类型订单分配正数量，D类型订单分配负数量
    order_df['EffectiveQty'] = np.where(order_df['Type'] == 'A',
                                        order_df['Qty'],
                                        -order_df['Qty'])

    # 按TICKER, Side, OrderNo分组，计算每个订单的净委托量
    net_orders = order_df.groupby(['TICKER', 'Side', 'OrderNo']).agg({
        'EffectiveQty': 'sum',
        'Price': 'first',
        'min': 'first',
        'TickTime': 'first'
    }).reset_index()
    # 过滤掉净委托量为0或负数的订单
    effective_orders = net_orders[net_orders['EffectiveQty'] > 0].copy()
    # 计算订单金额
    effective_orders['OrderAmount'] = effective_orders['EffectiveQty'] * effective_orders['Price']
    return effective_orders


def classify_orders_by_amount(effective_orders):
    """
    根据OrderAmount对订单进行分类
    """
    # 定义金额阈值
    thresholds = {
        'small': 40000,  # 4万
        'medium': 200000,  # 20万
        'large': 1000000  # 100万
    }

    orders_classified = effective_orders.copy()

    # 计算各个区间的条件
    small_condition = orders_classified['OrderAmount'] < thresholds['small']
    medium_condition = ((orders_classified['OrderAmount'] >= thresholds['small']) &
                        (orders_classified['OrderAmount'] < thresholds['medium']))
    large_condition = ((orders_classified['OrderAmount'] >= thresholds['medium']) &
                       (orders_classified['OrderAmount'] < thresholds['large']))
    xlarge_condition = orders_classified['OrderAmount'] >= thresholds['large']

    # 添加分类标签
    orders_classified['AmountCategory'] = ''
    orders_classified.loc[small_condition, 'AmountCategory'] = 'small'
    orders_classified.loc[medium_condition, 'AmountCategory'] = 'medium'
    orders_classified.loc[large_condition, 'AmountCategory'] = 'large'
    orders_classified.loc[xlarge_condition, 'AmountCategory'] = 'xlarge'

    return orders_classified


def get_trade_data_by_orders(trade_data, order_data):
    """
    根据订单数据匹配交易数据，获取实际成交量和成交额
    """
    print("开始匹配订单和交易数据...")

    # 确保交易数据有TradeMoney列
    if 'TradeMoney' not in trade_data.columns:
        trade_data = trade_data.copy()
        trade_data['TradeMoney'] = trade_data['Qty'] * trade_data['Price']
    else:
        trade_data = trade_data.copy()

    # 分别处理买单和卖单
    buy_orders = order_data[order_data['Side'] == 1].copy()
    sell_orders = order_data[order_data['Side'] == 2].copy()

    # print(f"买单数量: {len(buy_orders)}, 卖单数量: {len(sell_orders)}")

    # 处理买单的交易数据
    # print("处理买单交易数据...")
    buy_trade_agg = trade_data.groupby(['TICKER', 'BuyOrderNo']).agg({
        'TradeMoney': 'sum',
        'Qty': 'sum'
    }).reset_index()

    # 过滤有效的买单号
    buy_trade_agg = buy_trade_agg[buy_trade_agg['BuyOrderNo'] > 0]
    # print(f"买单交易聚合数据: {len(buy_trade_agg)}行")

    # 将买单订单数据与交易数据匹配
    buy_matched = pd.merge(
        buy_orders[['TICKER', 'OrderNo', 'AmountCategory']],
        buy_trade_agg,
        left_on=['TICKER', 'OrderNo'],
        right_on=['TICKER', 'BuyOrderNo'],
        how='inner'
    )
    buy_matched['Side'] = 'buy'
    # print(f"买单匹配成功: {len(buy_matched)}行")

    # 处理卖单的交易数据
    # print("处理卖单交易数据...")
    sell_trade_agg = trade_data.groupby(['TICKER', 'SellOrderNo']).agg({
        'TradeMoney': 'sum',
        'Qty': 'sum'
    }).reset_index()

    # 过滤有效的卖单号
    sell_trade_agg = sell_trade_agg[sell_trade_agg['SellOrderNo'] > 0]
    # print(f"卖单交易聚合数据: {len(sell_trade_agg)}行")

    # 将卖单订单数据与交易数据匹配
    sell_matched = pd.merge(
        sell_orders[['TICKER', 'OrderNo', 'AmountCategory']],
        sell_trade_agg,
        left_on=['TICKER', 'OrderNo'],
        right_on=['TICKER', 'SellOrderNo'],
        how='inner'
    )
    sell_matched['Side'] = 'sell'
    # print(f"卖单匹配成功: {len(sell_matched)}行")

    # 合并买单和卖单数据
    all_matched = pd.concat([
        buy_matched[['TICKER', 'AmountCategory', 'Side', 'TradeMoney', 'Qty']],
        sell_matched[['TICKER', 'AmountCategory', 'Side', 'TradeMoney', 'Qty']]
    ], ignore_index=True)

    # print(f"总匹配数据: {len(all_matched)}行")

    return all_matched


def calculate_stock_level_summary(matched_data, date, output_path):
    """
    计算每只股票的汇总数据：每只股票×买卖双方×小中大超大单×成交量和成交额
    """
    # print("计算每只股票的汇总数据...")

    # 按股票、买卖方向和金额分类聚合
    summary = matched_data.groupby(['TICKER', 'Side', 'AmountCategory']).agg({
        'Qty': 'sum',
        'TradeMoney': 'sum'
    }).reset_index()

    summary_buy = summary[summary['Side'] == 'buy'][['TICKER', 'AmountCategory', 'Qty', 'TradeMoney']]
    summary_sell = summary[summary['Side'] == 'sell'][['TICKER', 'AmountCategory', 'Qty', 'TradeMoney']]
    # print(f"聚合后数据行数: {len(summary)}")

    # 获取所有股票列表
    all_tickers = summary['TICKER'].unique()
    # print(f"股票数量: {len(all_tickers)}")

    summary_buy_qty = summary_buy.pivot_table(columns=['AmountCategory'], index=['TICKER'], values='Qty')
    summary_buy_amount = summary_buy.pivot_table(columns=['AmountCategory'], index=['TICKER'], values='TradeMoney')

    summary_buy_qty = summary_buy_qty.reset_index(drop=False)
    summary_buy_amount = summary_buy_amount.reset_index(drop=False)
    summary_buy_all = pd.merge(summary_buy_qty, summary_buy_amount, how='outer', on=['TICKER'],
                               suffixes=('_buy_qty', '_buy_amount'))

    summary_sell_qty = summary_sell.pivot_table(columns=['AmountCategory'], index=['TICKER'], values='Qty')
    summary_sell_amount = summary_sell.pivot_table(columns=['AmountCategory'], index=['TICKER'], values='TradeMoney')

    summary_sell_qty = summary_sell_qty.reset_index(drop=False)
    summary_sell_amount = summary_sell_amount.reset_index(drop=False)
    summary_sell_all = pd.merge(summary_sell_qty, summary_sell_amount, how='outer', on=['TICKER'],
                                suffixes=('_sell_qty', '_sell_amount'))

    summary_all = pd.merge(summary_sell_all, summary_buy_all, how='outer', on=['TICKER'])
    summary_all = summary_all.fillna(0)
    summary_all.index.name = None
    summary_all['DATE'] = date
    summary_all = summary_all.reset_index(drop=True)
    summary_all.columns.name = None
    feather.write_dataframe(summary_all, os.path.join(output_path, f'{date}.feather'))


def process_daily_moneyflow(date, output_path):
    print(date)
    order_file_path = rf'\\Desktop-79nue61\sh\{date}_order_sh.feather'
    trade_file_path = rf'\\Desktop-79nue61\sh\{date}_trade_sh.feather'
    """
    处理每日资金流向数据的主函数
    """
    start_time = time.time()
    # 1. 读取和处理订单数据
    sh_order_data = pd.read_feather(order_file_path)
    # 2. 计算有效订单
    effective_orders = calculate_effective_orders_corrected(sh_order_data)
    # print(f"有效订单数: {len(effective_orders)}")
    # 3. 按OrderAmount分类订单
    orders_classified = classify_orders_by_amount(effective_orders)
    # 4. 读取交易数据
    sh_trade_data = pd.read_feather(trade_file_path)
    # print(f"交易数据形状: {sh_trade_data.shape}")
    # 5. 匹配订单和交易数据
    matched_data = get_trade_data_by_orders(sh_trade_data, orders_classified)
    # 6. 计算每只股票的汇总
    calculate_stock_level_summary(matched_data, date, output_path=output_path)
    print(f"总耗时: {time.time() - start_time:.2f}秒")


if __name__ == "__main__":
    all_date_sh = os.listdir(r'\\DESKTOP-79NUE61\sh')
    all_date_sh = [i[:8] for i in all_date_sh]
    all_date_sh = list(set(all_date_sh))
    all_date_sh.sort()
    os.makedirs(r'\\DESKTOP-79NUE61\money_flow_sh',exist_ok=True)
    for d in all_date_sh:
        if d >= '20241001':
            if not os.path.exists(rf'\\DESKTOP-79NUE61\money_flow_sh\{d}.feather'):
                process_daily_moneyflow(d, r'\\DESKTOP-79NUE61\money_flow_sh')