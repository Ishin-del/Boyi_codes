import datetime
import os
import time
import warnings

import pandas as pd

from tool_tyx.path_data import DataPath
from factor_codes.更新代码 import attention_factor, ctr_factor_final, fear_factor_final, haitong58_916, \
    order_trade_factor, trade_imbalance, rewrite_traction_LUD, rewrite2_traction_LUD, \
    traction_LUD, SSF_V2, risk_uncer, 激流勇进, 暗流涌动, factor_min_data, haitong59, ideal_amplitude_factor, \
    rewrite_左尾风险, valid_order_100s, vol_in_day_v2, vol_ratio_on_price_volume,ret_on_act_diff,UMR_v1,UMR_v2,\
    turnTop4Sum,factor_on_high_freq3
from 更新code import (GTR日频,
                     GUD,
                     HIgh_Freq_RSkew,
                     MSF_weight_decay,
                     SSF_weight_decay,
                     一视同仁,
                     勇攀高峰,
                     左尾风险,
                     随波逐流)

from test_tools import 多进程_增量中性化

def run_factors(today):
    GTR日频.update(today)
    GUD.update(today)
    HIgh_Freq_RSkew.update(today)
    MSF_weight_decay.update(today)
    SSF_weight_decay.update(today)
    一视同仁.update(today)
    勇攀高峰.update(today)
    左尾风险.update(today)
    随波逐流.update(today)
    # ------------------------------------
    haitong59.update(today)
    ctr_factor_final.update(today)
    fear_factor_final.update(today)
    ideal_amplitude_factor.update(today)
    rewrite_左尾风险.update(today)
    SSF_V2.update(today)
    暗流涌动.update(today)
    vol_in_day_v2.update(today)
    factor_min_data.update(today)
    激流勇进.update(today)
    order_trade_factor.update(today)
    risk_uncer.update(today)
    valid_order_100s.update(today)
    haitong58_916.update(today)
    traction_LUD.update(today)
    rewrite_traction_LUD.update(today)
    rewrite2_traction_LUD.update(today)
    vol_ratio_on_price_volume.update(today)
    ret_on_act_diff.update(today)
    UMR_v1.update(today)
    UMR_v2.update(today)
    turnTop4Sum.update(today)
    attention_factor.update(today)
    trade_imbalance.update(today)
    factor_on_high_freq3.update(today)

def update_factors():
    warnings.filterwarnings('ignore')
    today = datetime.datetime.today().strftime('%Y%m%d')
    # today='20251103'
    while 1:
        calendar = pd.read_csv(os.path.join(DataPath.to_path, 'calendar.csv'))
        flag1=int(today) in calendar.trade_date.unique()
        flag2=today in os.listdir(r'\\Desktop-79nue61\更新完成标识\daily')
        flag3=today in os.listdir(r'\\Desktop-79nue61\更新完成标识\tick')
        if flag1 and flag2 and flag3:
            run_factors(today)
            os.makedirs(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\更新完成标识\原始数据完成标识\拟使用量价因子\{today}',exist_ok=True)
            多进程_增量中性化.signal_processor_terminal(today)
            os.makedirs(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\更新完成标识\中性化完成标识\{today}',exist_ok=True)
            break
        else:
            time.sleep(300)
            print(f'{datetime.datetime.now()}无法更新')


if __name__ == '__main__':
    # if True:
    #     update_factors()
    import schedule
    schedule.every().day.at('23:00').do(update_factors)
    while True:
        schedule.run_pending()
        time.sleep(30)


