import pandas as pd
import numpy as np
import os
import time
import warnings


def calculate_effective_orders_corrected(order_data):
    """
    计算有效的委托订单
    先按订单号计算净委托量，再分配到分钟
    """
    order_df = order_data.copy()

    # 为买卖方向创建统一的订单号列
    order_df['OrderNo'] = order_df['TickIndex']

    # 为A类型订单分配正数量，D类型订单分配负数量
    order_df['EffectiveQty'] = np.where(order_df['Type'] == 'A',
                                        order_df['Qty'],
                                        -order_df['Qty'])

    # 第一步：按TICKER, Side, OrderNo分组，计算每个订单的净委托量
    net_orders = order_df.groupby(['TICKER', 'Side', 'OrderNo']).agg({
        'EffectiveQty': 'sum',  # 计算净委托量
        'Price': 'first',  # 取第一个价格
        'min': 'first',  # 取第一次出现的分钟作为该订单的分钟
        'TickTime': 'first'  # 取第一次出现的时间作为该订单的时间
    }).reset_index()

    # 第二步：过滤掉净委托量为0或负数的订单
    effective_orders = net_orders[net_orders['EffectiveQty'] > 0]

    return effective_orders


def calculate_percentiles_and_intervals(data, qty_column, prefix, value_column_name):
    warnings.filterwarnings(action='ignore')
    """
    通用函数：计算分位数并分配区间

    Parameters:
    - data: 输入数据
    - qty_column: 用于计算分位数的列名
    - prefix: 区间列名前缀 (如 'order' 或 'trade')
    - value_column_name: 分位数值列名 (如 'EffectiveQty_Value' 或 'Qty_Value')
    """
    # 1. 计算分位数
    percentiles = data.groupby('TICKER')[qty_column].quantile([0.2, 0.4, 0.6, 0.8, 1.0]).unstack()
    percentiles.columns = ['p20', 'p40', 'p60', 'p80', 'p100']
    percentiles = percentiles.reset_index()

    # 2. 转换为长格式的分位数表
    percentiles_df = pd.melt(
        percentiles,
        id_vars=['TICKER'],
        value_vars=['p20', 'p40', 'p60', 'p80', 'p100'],
        var_name='Percentile_Type',
        value_name=value_column_name
    )

    # 映射百分位数
    percentile_mapping = {'p20': 20.0, 'p40': 40.0, 'p60': 60.0, 'p80': 80.0, 'p100': 100.0}
    percentiles_df['Percentile'] = percentiles_df['Percentile_Type'].map(percentile_mapping)
    percentiles_df = percentiles_df[['TICKER', 'Percentile', value_column_name]].sort_values(['TICKER', 'Percentile'])

    # 3. 将分位数信息合并到原数据
    data_with_percentiles = data.merge(
        percentiles[['TICKER', 'p20', 'p40', 'p60', 'p80']],
        on='TICKER'
    )
    # 4. 使用向量化操作计算区间
    data_with_percentiles[f'{prefix}_1_num'] = (data_with_percentiles[qty_column] < data_with_percentiles['p20']).astype(int)
    data_with_percentiles[f'{prefix}_2_num'] = ((data_with_percentiles[qty_column] >= data_with_percentiles['p20']) &
                                           (data_with_percentiles[qty_column] < data_with_percentiles['p40'])).astype(
        int)
    data_with_percentiles[f'{prefix}_3_num'] = ((data_with_percentiles[qty_column] >= data_with_percentiles['p40']) &
                                           (data_with_percentiles[qty_column] < data_with_percentiles['p60'])).astype(
        int)
    data_with_percentiles[f'{prefix}_4_num'] = ((data_with_percentiles[qty_column] >= data_with_percentiles['p60']) &
                                           (data_with_percentiles[qty_column] < data_with_percentiles['p80'])).astype(
        int)
    data_with_percentiles[f'{prefix}_5_num'] = (data_with_percentiles[qty_column] >= data_with_percentiles['p80']).astype(
        int)

    # 5. 按分钟和股票聚合
    if prefix == 'order':
        buy = data_with_percentiles[data_with_percentiles['Side'] == 1]
        sell = data_with_percentiles[data_with_percentiles['Side'] == 2]
        buy['amt'] = buy['EffectiveQty']*buy['Price']
        sell['amt'] = sell['EffectiveQty']*sell['Price']
        for i in range(5):
            buy[f'EffectiveQty_{i+1}'] = buy['EffectiveQty']*buy[f'{prefix}_{i+1}_num']
            buy[f'Amount_{i+1}'] = buy['amt']*buy[f'{prefix}_{i+1}_num']
            sell[f'EffectiveQty_{i+1}'] = sell['EffectiveQty']*sell[f'{prefix}_{i+1}_num']
            sell[f'Amount_{i+1}'] = sell['amt']*sell[f'{prefix}_{i+1}_num']

        minute_data_buy = buy.groupby(['min', 'TICKER'])[
            [f'EffectiveQty_1', f'EffectiveQty_2', f'EffectiveQty_3', f'EffectiveQty_4', f'EffectiveQty_5',
             f'Amount_1', f'Amount_2', f'Amount_3', f'Amount_4', f'Amount_5',
             f'{prefix}_1_num', f'{prefix}_2_num', f'{prefix}_3_num', f'{prefix}_4_num', f'{prefix}_5_num']
        ].sum().reset_index()

        minute_data_sell = sell.groupby(['min', 'TICKER'])[
            [f'EffectiveQty_1', f'EffectiveQty_2', f'EffectiveQty_3', f'EffectiveQty_4', f'EffectiveQty_5',
             f'Amount_1', f'Amount_2', f'Amount_3', f'Amount_4', f'Amount_5',
             f'{prefix}_1_num', f'{prefix}_2_num', f'{prefix}_3_num', f'{prefix}_4_num', f'{prefix}_5_num']
        ].sum().reset_index()
        minute_data = pd.merge(minute_data_sell,minute_data_buy,how='outer',on=['TICKER','min'],suffixes=('_order_sell','_order_buy'))
        minute_data = minute_data.fillna(0)

    else:
        for i in range(5):
            data_with_percentiles[f'EffectiveQty_{i+1}'] = data_with_percentiles['Qty']*data_with_percentiles[f'{prefix}_{i+1}_num']
            data_with_percentiles[f'Amount_{i+1}'] = data_with_percentiles['TradeMoney']*data_with_percentiles[f'{prefix}_{i+1}_num']

        minute_data = data_with_percentiles.groupby(['min', 'TICKER'])[
            [f'EffectiveQty_1', f'EffectiveQty_2', f'EffectiveQty_3', f'EffectiveQty_4', f'EffectiveQty_5',
             f'Amount_1', f'Amount_2', f'Amount_3', f'Amount_4', f'Amount_5',
             f'{prefix}_1_num', f'{prefix}_2_num', f'{prefix}_3_num', f'{prefix}_4_num', f'{prefix}_5_num']
        ].sum().reset_index()

    return percentiles_df, minute_data


def next_minute(time_val):
    """计算下一分钟的时间值，正确处理时间进位"""
    hour = time_val // 100
    minute = time_val % 100

    minute += 1
    if minute == 60:
        minute = 0
        hour += 1

    return hour * 100 + minute


def process_order_data(order_file_path):
    """处理订单数据的主函数"""
    # 读取订单数据
    sh_order_data = pd.read_feather(order_file_path)
    sh_order_data['min'] = (sh_order_data['TickTime'].astype(str).str[-9:-5]).astype('int32')

    # 提取日期
    # date = sh_order_data['TickTime'].astype(str).str[:8].iloc[0]

    # sh_continue_order_data = sh_order_data[(sh_order_data['min'] >= 930) & (sh_order_data['min'] <= 1459)]

    # 计算有效订单
    effective_orders = calculate_effective_orders_corrected(sh_order_data)
    # 使用通用函数计算分位数和分钟订单
    percentiles_df, minute_orders = calculate_percentiles_and_intervals(
        effective_orders, 'EffectiveQty', 'order', 'EffectiveQty_Value'
    )

    # 处理集合竞价时间段的数据
    minute_orders_30 = minute_orders[(minute_orders['min'] >= 915) & (minute_orders['min'] < 930)]
    minute_orders_30_df = minute_orders_30.groupby(['TICKER'])[
        [i for i in minute_orders_30.columns if i not in ['TICKER', 'min']]].sum()
    minute_orders_30_df.reset_index(inplace=True)
    minute_orders_30_df.insert(0, 'min', 930)

    # 调整时间
    condition = (minute_orders['min'] >= 930) & (minute_orders['min'] <= 1129)
    minute_orders.loc[condition, 'min'] = minute_orders.loc[condition, 'min'].apply(next_minute).astype('int32')
    minute_orders = pd.concat([minute_orders[minute_orders['min'] > 930], minute_orders_30_df], axis=0)
    minute_orders.sort_values(by=['min', 'TICKER'], inplace=True)
    minute_orders.reset_index(drop=True, inplace=True)

    return minute_orders, percentiles_df


def process_trade_data(trade_file_path):
    """处理交易数据的主函数"""
    # 读取交易数据
    sh_trade_data = pd.read_feather(trade_file_path)
    sh_trade_data['min'] = (sh_trade_data['TickTime'].astype(str).str[:-5]).astype('int64')

    # 提取日期

    # 使用通用函数计算分位数和分钟交易
    percentiles_df, minute_trades = calculate_percentiles_and_intervals(
        sh_trade_data, 'Qty', 'trade', 'Qty_Value'
    )

    # 调整时间
    condition = (minute_trades['min'] >= 930) & (minute_trades['min'] <= 1129)
    minute_trades.loc[condition, 'min'] = minute_trades.loc[condition, 'min'].apply(next_minute).astype('int64')
    minute_trades.loc[minute_trades['min'] == 925, 'min'] = 930

    return minute_trades, percentiles_df


def main_sh(date):
    """主函数"""
    start_time = time.time()

    # 文件路径
    order_file_path = rf'\\Desktop-79nue61\sh\{date}_order_sh.feather'
    trade_file_path = rf'\\Desktop-79nue61\sh\{date}_trade_sh.feather'

    print("=== 开始处理数据 ===")

    # 处理订单数据
    print("正在处理订单数据...")
    minute_orders, order_percentiles = process_order_data(order_file_path)
    # print(f"订单数据处理完成，共{len(minute_orders)}行")

    # 处理交易数据
    print("正在处理交易数据...")
    minute_trades, trade_percentiles = process_trade_data(trade_file_path)
    # print(f"交易数据处理完成，共{len(minute_trades)}行")

    # 确保日期一致
    # date = order_date if order_date == trade_date else order_date
    print(f"数据日期: {date}")

    # 合并数据
    print("正在合并数据...")
    min_trade_order = pd.merge(minute_orders, minute_trades, on=['min', 'TICKER'], how='outer').fillna(0)
    # 保存结果
    end_time = time.time()
    print(f"\n=== 处理完成 ===")
    print(f"总耗时: {end_time - start_time:.2f}秒")
    print(f"最终数据形状: {min_trade_order.shape}")
    return min_trade_order
    # return minute_orders, minute_trades, order_percentiles, trade_percentiles, date


if __name__ == "__main__":
    import feather

    all_date_sh = os.listdir(r'\\Desktop-79nue61\sh')
    os.makedirs(r'\\Desktop-79nue61\SH_min_data_big_order', exist_ok=True)
    all_date_sh = [i[:8] for i in all_date_sh]
    all_date_sh = list(set(all_date_sh))
    all_date_sh = [i for i in all_date_sh if i < '20220101']
    all_date_sh.sort()
    for d in all_date_sh:
        if not os.path.exists(rf'\\DESKTOP-79NUE61\SH_min_data_big_order\{d}.feather'):
            result = main_sh(d)
            feather.write_dataframe(result, rf'\\DESKTOP-79NUE61\SH_min_data_big_order\{d}.feather')

    # minute_orders, minute_trades, order_percentiles, trade_percentiles, date = main_sh()