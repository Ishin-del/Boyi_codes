# -*- coding: utf-8 -*-
# Created on  2022/8/23 9:10
# @Author: Yangyu Che
# @File: 前后数据对比.py
# @Contact: cheyangyu@126.com
# @Software: PyCharm

import os
import feather
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm

new_data_path = r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\2021.6-2025.8'
old_data_path = r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\2020-2022.12'

#
old_file_ls = os.listdir(old_data_path)
old_file_ls = [i for i in old_file_ls if not ('汇总' in i or '用于更新' in i)]
new_file_ls = old_file_ls
# old_file_ls = ['fin_factor_20230701_base2.feather']
# new_file_ls = ['fin_factor_20230801_base2.feather']
# old_file_ls = new_file_ls
for i in range(len(old_file_ls)):
    # print(file_name)
    old_name = old_file_ls[i]
    new_name = new_file_ls[i]

    # df_daily = feather.read_dataframe(r'E:\QuantWork\Data_Storage\Factor_Storage\daily.feather')
    df_old = feather.read_dataframe(os.path.join(old_data_path, old_name))
    # df_old = df_old[df_old['DATE'] <= '20210630']
    df_new = feather.read_dataframe(os.path.join(new_data_path, new_name))
    print('old:', df_old['DATE'].max(), 'new:', df_new['DATE'].max())
    df_new = df_new[df_new['DATE'] <= df_old['DATE'].max()]

    df_old['sig'] = 0
    df_new['sig'] = 1

    inter_columns = np.intersect1d(df_old.columns, df_new.columns)
    indi_col = 'TICKER' if 'TICKER' in inter_columns else 'commodity'
    inter_columns = np.setdiff1d(inter_columns, [indi_col, 'end_date', 'DATE', 'sig'])

    df_diff = pd.merge(df_old[[indi_col, 'DATE', 'sig']],
                       df_new[[indi_col, 'DATE', 'sig']],
                       on=[indi_col, 'DATE'],
                       how='outer', suffixes=(None, '_new'))
    df_new_miss = df_diff[df_diff['sig_new'].isna()]
    df_new_add = df_diff[df_diff['sig'].isna()]
    # 首先检查之前有因子值的股票，新的因子值和原来是否有区别
    df_old_check = pd.DataFrame()
    for col in tqdm(inter_columns, f'检查已有数据股票新旧因子值:{old_file_ls[i]}'):
        df_diff = pd.merge(df_old[[indi_col, 'DATE', col]],
                           df_new[[indi_col, 'DATE', col]],
                           on=[indi_col, 'DATE'],
                           how='outer', suffixes=(None, '_new'))
        df_value_check = df_diff[~np.isclose(df_diff[col], df_diff[col + '_new'], equal_nan=True)]
        # df_value_check = df_diff[df_diff[col]!=df_diff[col + '_new']].dropna(subset=[col,col + '_new'],how='all')
        if df_new_miss.empty and df_new_add.empty and df_value_check.empty:
            pass
        else:
            for j, df in [('new_miss', df_new_miss), ('new_add', df_new_add),
                          ('diff_value', df_value_check)]:
                if not df.empty:
                    print(j, ': ', df, df['DATE'].max())
