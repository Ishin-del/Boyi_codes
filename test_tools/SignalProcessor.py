from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import pandas as pd
import feather
import datetime as dt
import statsmodels.api as sm
from tqdm import tqdm
import os
from statsmodels.stats.weightstats import DescrStatsW
from scipy.special import erfinv
from sklearn.linear_model import LinearRegression
import sys

sys.path.append('../')
from tool_tyx.data_path import FactorPath


def read_file(path, file_name, file_type=None):
    file_type = file_type or file_name[file_name.index('.'):]

    if 'xlsx' in file_type:
        df = pd.read_excel(os.path.join(path, file_name))
        try:
            df['DATE'] = df['DATE'].map(lambda x: x.strftime('%Y%m%d'))
        except:
            try:
                df['DATE'] = df['DATE'].astype(str)
            except:
                pass
    elif 'csv' in file_type:
        df = pd.read_csv(os.path.join(path, file_name), dtype={'DATE': str})
    elif 'feather' in file_name:
        df = feather.read_dataframe(os.path.join(path, file_name))
    else:
        raise TypeError('The type of file is incorrect.')

    return df


class SignalProcessor(object):
    """> class SignalProcessor(object)
        信号处理类，整合多种信号处理函数。
        Examples:
            processor = SigalProcessor()
            processed = processor(signal_series)
        """

    def __init__(self, fillna, win, stand, dropstyle):
        """> __init__(self, fillna, win, neutr, stand, dropstyle, win_type, n_draw, pvalue)
                Inputs:
                    fillna: [bool] 是否用行业中位数填充缺失值
                    win: [bool] 是否去极值
                    neutr: [bool] 是否行业中性化
                    stand: [bool] 是否标准化
                    win_type: [str] 去极值方式，可选'NormDistDraw'/'QuantileDraw'
                    n_draw: [int] 正态分布去极值的迭代次数
                    pvalue: [float] 分位数去极值的分位数指定，处理头尾合计pvalue的异常
                    nor_weight: [None, str or series] 因子标准化时的权重，None则为无权重，
                                str表示权重类型（一般为'cap'，表示用市值进行加权），series则表示直接输入权重序列
                    fillna_type: [str] 表示在填充缺失值时的方式，可选 'median'/'mean'/'regression'
                    dropstyle_mode: [int] 剔除风格因子（做回归）的模式（待补充）
                        0: y = β行业 + ε
                        1: y = β1行业 + β2市值 + ε
                        2: y = β1行业 + β2市值 + β3市值**2 +β4市值**3 +ε
                        3: y = β1行业 + β2市值 + β3行业 * 市值 +ε
                        4: y = β1行业 + β2市值 + β3市值**2 +β4市值**3 + β5行业 * 市值 + ε
                """
        self.fillna = fillna
        self.win = win
        self.stand = stand
        self.dropstyle = dropstyle

    def __repr__(self):
        s = 'SignalProcessor(%s)' % ', '.join(map(lambda x: '%s=%s' % x, self.__dict__.items()))
        # self.__dict__.items()表示self这个实例的属性元组对列表，例如:
        # [('fillna', True),('win', True),('stand', True),('win_type', 'NormDistDraw'),('n_draw', 5),('pvalue', .05)]
        return s  # s = SignalProcessor(fillna=True, win=True, stand=True, win_type=NormDistDraw, n_draw=5, pvalue=05)

    __str__ = __repr__

    def __call__(self, raw_data, cap, stk_indus, stand_type='Zscore', fillna_type='median_local',
                 win_type='NormDistDraw',
                 dropstyle_mode=1,
                 n_draw=1, pvalue=.05, stand_weight=None, cap_mode='normal', regress_mode='OLS',
                 process_order=None):

        if len(raw_data) == 1 or len(raw_data.dropna()) <= 1:
            return raw_data
        assert stand_type.upper() in ['ZSCORE', 'RANK_GAUSS', 'RANK']

        self.win_type = win_type
        self.n_draw = n_draw
        self.pvalue = pvalue
        self.dropstyle_mode = dropstyle_mode
        self.stand_weight = stand_weight
        self.fillna_type = fillna_type
        self.stand_type = stand_type

        processed = raw_data.copy()

        if process_order != None:
            # 按照指定顺序进行Signal Process
            for procedure in process_order:
                if procedure == 'fillna':
                    processed = self.signal_fill(processed, stk_indus, cap)
                if procedure == 'win':
                    processed = self.my_winsorize(processed)
                if procedure == 'stand':
                    processed = self.my_standardize(processed, cap)
                if procedure == 'dropstyle':
                    processed = self.my_dropstyle(processed, cap, stk_indus, cap_mode=cap_mode,
                                                  regress_mode=regress_mode)
        else:
            if self.fillna:  # 填充缺失值
                processed = self.signal_fill(processed, stk_indus, cap)
            if self.win:  # 去极值
                processed = self.my_winsorize(processed)
            if self.stand:  # 标准化
                processed = self.my_standardize(processed, cap)
            if self.dropstyle:  # 剔除行业、市值的影响
                processed = self.my_dropstyle(processed, cap, stk_indus, cap_mode=cap_mode,
                                              regress_mode=regress_mode)

        processed.name = raw_data.name
        return processed.reset_index(drop=False)

    def _fillna_industry_sklearn(self, series, cap):
        '''
        内置函数：当fillna_type为regression时，用此方法对缺失值进行填充
        【注意】：使用此方法时，一定要先进行去极值和标准化
        '''

        series_na = series[series.isna()]
        series_num = series[~series.isna()]

        if len(series_na) > len(series) * 0.4:
            print('缺失值超过40%, 不做处理')
            return series
        elif len(series_na) == 0:
            return series
        else:
            line_reg = LinearRegression()
            line_reg.fit(np.log(cap[series_num.index]).to_numpy().reshape(-1, 1), series_num.to_numpy().reshape(-1, 1))
            series_fill = line_reg.predict(np.log(cap[series_na.index]).to_numpy().reshape(-1, 1))
            series_fill = pd.Series(series_fill.reshape(-1), index=series_na.index)
            return series_num.append(series_fill)

    def signal_fill(self, raw_data, stk_indus, cap, max_nan=.2, fillna_type=None):
        """
        > signal_fill(raw_data, stk_indus=None, max_nan=.2, fillna_type = None)
        将输入中的缺失值用行业中位数进行填充。
        Inputs:
            raw_data: [dict/pd.Series/np.array] 输入待处理数据
            stk_indus: [pd.Series] index为股票，value为所属行业
            max_nan: [float] 如果NaN比例大于该值，则作废该条信号
            fillna_type: 在__init__中有解释，表示填充方法
        Returns:
            processed: 填充后的序列，类型与输入相同
        """
        fillna_type = fillna_type or self.fillna_type
        nan_pct = 1 - raw_data.count() / len(raw_data)
        series = raw_data if isinstance(raw_data, pd.Series) else pd.Series(raw_data)

        if nan_pct > max_nan:
            processed = raw_data
            return processed
        else:
            pass

        if '_' in fillna_type:
            global_type = fillna_type.split('_')[1]
            fillna_type = fillna_type.split('_')[0]
        else:
            global_type = None

        if fillna_type == 'median' or fillna_type == 'mean':
            # 按照行业代码分组，indus_fill索引为行业代码，内容为行业填充值
            indus_narate = series.groupby(by=stk_indus).apply(lambda x: 1 - x.count() / len(x))
            nan_idx = series[series.isnull()].index.to_list()
            if fillna_type == 'median':
                indus_fill = series.dropna().groupby(by=stk_indus).apply(np.median)

                if indus_fill.index.shape[0] != indus_narate.index.shape[0]:
                    indus_fill_append = pd.Series(np.nan, index=np.setdiff1d(indus_narate.index, indus_fill.index))
                    indus_fill = indus_fill.append(indus_fill_append)
                else:
                    pass

                total_fill = np.median(series.dropna())
            else:
                indus_fill = series.dropna().groupby(by=stk_indus).apply(np.mean)

                if indus_fill.index.shape[0] != indus_narate.index.shape[0]:
                    indus_fill[np.setdiff1d(indus_narate.index, indus_fill.index)[0]] = np.nan
                else:
                    pass
                total_fill = np.mean(series.dropna())

            processed = series.fillna(stk_indus[nan_idx].dropna().apply(lambda x: indus_fill[x] if
            np.isfinite(indus_fill[x].astype(float)) and indus_narate[x] <= max_nan else np.nan).to_dict())

            if global_type == 'global':
                processed = processed.fillna(total_fill)
                # 如果行业中位数不是空值就用行业中位填充series中的空值，如果行业中位数是空值，就用市场中位数填充series中的空值
            else:
                pass
        else:
            # fillna_type = 'regression'
            processed = series.groupby(by=stk_indus).apply(self._fillna_industry_sklearn, cap=cap).reset_index(
                level='Citic_Code', drop=True)

        if isinstance(raw_data, dict):
            processed = processed.to_dict()  # 如果raw_data是字典类型，则将processed转换成字典型
        else:
            pass

        return processed

    def my_winsorize(self, raw_data, n_draw=None, pvalue=None, win_type=None):
        """> my_winsorize(raw_data, win_type='NormDistDraw', n_draw=5, pvalue=.05)
        去极值函数
        Inputs:
            raw_data: [dict/pd.Series/np.array] 输入待处理数据
            win_type: [string] 'NormDistDraw'/'QuantileDraw'
            n_draw: [int] 正态分布去极值的迭代次数
            pvalue: [float] 分位数去极值的分位数指定，处理头尾合计pvalue的异常
        Returns:
            processed: 去极值后的结果，类型与输入相同
        """
        n_draw = n_draw or self.n_draw
        pvalue = pvalue or self.pvalue
        win_type = win_type or self.win_type
        series = raw_data if isinstance(raw_data, pd.Series) else pd.Series(raw_data)

        processed = series.copy()  # 对series进行浅copy，即只copy父对象，不copy子对象
        if win_type == 'QuantileDraw':  # 如果去极值的方式为QuantileDraw

            if pvalue <= 0 or pvalue >= 1:  # 如果pvalue<0或>1，则提示'pvalue' should lie in (0, 1).
                raise ValueError("'pvalue' should lie in (0, 1).")

            left = np.percentile(series, int(pvalue * 50))  # 表示小于l的观察值占pvalue * 50个百分比
            right = np.percentile(series, int(100 - pvalue * 50))  # 表示大于r的观察值占pvalue * 50个百分比
            left_limit = np.percentile(series, int(0.04 * 50))
            right_limit = np.percentile(series, int(100 - 0.04 * 50))

            if np.isclose(left - left_limit, 0.0):
                processed[processed < left] = left  # 小于左侧分位数l的用左侧分位数l填充
            else:
                limit = np.abs(left_limit - left)
                processed[processed < left] = left - limit * (left - processed[processed < left]) / (
                        left - processed[processed < left]).max()

            if np.isclose(right_limit - right, 0.0):
                processed[processed < left] = left  # 小于左侧分位数l的用左侧分位数l填充
            else:
                limit = np.abs(right_limit - right)
                processed[processed > right] = right + limit * (processed[processed > right] - right) / (
                        processed[processed > right] - right).max()

        elif win_type == 'MAD':

            median = series.median()
            MAD = np.abs(series - median).median()

            left = median - 3 * 1.483 * MAD
            right = median + 3 * 1.483 * MAD
            left_limit = median - 3.5 * 1.483 * MAD
            right_limit = median + 3.5 * 1.483 * MAD

            if np.isclose(left - left_limit, 0.0):
                processed[processed < left] = left
            else:
                limit = np.abs(left_limit - left)
                processed[processed < left] = left - limit * (left - processed[processed < left]) / (
                        left - processed[processed < left]).max()

            if np.isclose(right_limit - right, 0.0):
                processed[processed > right] = right
            else:
                limit = np.abs(right_limit - right)
                processed[processed > right] = right + limit * (processed[processed > right] - right) / (
                        processed[processed > right] - right).max()

        elif 'NormDistDraw' in win_type:  # 如果去极值的方式为NormDistDraw，3-sigma
            if n_draw <= 0 or not isinstance(n_draw, int):
                # 如果迭代次数n_draw<=0或者不为整数，则报错'n_draw' should be a positive integer.
                raise ValueError("'n_draw' should be a positive integer.")
            for _ in range(n_draw):  # 循环迭代次数这么多次
                mean, std = processed.mean(), processed.std()
                lower, upper = mean - 3 * std, mean + 3 * std
                l_idx, u_idx = processed < lower, processed > upper

                if float(processed[l_idx].sum() + processed[u_idx].sum()) == 0:
                    break
                if 'zip' in win_type:
                    limit = 0.5 * std
                    processed[u_idx] = upper + limit * (processed[u_idx] - upper) / (
                            processed[u_idx] - upper).max()
                    processed[l_idx] = lower - limit * (lower - processed[l_idx]) / (
                            lower - processed[l_idx]).max()
                elif 'fill' in win_type:
                    processed[l_idx] = lower
                    processed[u_idx] = upper

        else:  # 如果去极值的方式异常，报错Unknown 'win_type'
            raise ValueError("Unknown 'win_type'")

        if isinstance(raw_data, dict):
            processed = processed.to_dict()  # 如果raw_data是字典类型，则将processed转换成字典型

        return processed  # 去极值后的序列，类型与输入相同

    def my_standardize(self, raw_data, cap, stand_weight=None, stand_type=None):

        """> my_standardize(raw_data)
        将输入标准化为0均值1方差
        Inputs:
            raw_data: [dict/pd.Series/np.array] 输入待处理数据
        Returns:
            processed: 0均值1方差的序列，类型与输入相同
        """
        stand_weight = stand_weight or self.stand_weight
        stand_type = stand_type or self.stand_type
        series = raw_data if isinstance(raw_data, pd.Series) else pd.Series(raw_data)

        if abs(series).max() == np.inf:
            raise ValueError('Data contains inf value. Please check.')

        series_nan = series[series.isnull()]
        series = series[~series.isnull()]

        if stand_type.upper() == 'ZSCORE':

            if stand_weight != None:
                mv_value = cap[series.index]
                weights = mv_value / mv_value.sum()

                series_weighted = DescrStatsW(np.array(series), weights=weights)
                weighted_mean = series_weighted.mean
                series = (series - weighted_mean) / series_weighted.std
                processed = series.append(series_nan)
            else:
                mean, std = series.mean(), series.std()  # 求series的均值和标准差
                if np.isclose(std, .0):
                    # 如果absolute(std - 0) <= (atol + rtol * absolute(0)), rtol=1e-05, atol=1e-08 ，即基本不存在标准差
                    std = 1
                    processed = (series - mean) / std
                else:
                    processed = (series - mean) / std  # 否则进行标准化
                processed = processed._append(series_nan).sort_index()

        elif stand_type.upper() == 'RANK_GAUSS':
            x = np.array(series)
            x = x.argsort().argsort()  # rank
            x = (x / x.max() - 0.5) * 2  # scale
            x = np.clip(x, -1 + 1e-6, 1 - 1e-6)
            processed = pd.Series(erfinv(x), index=series.index)
        else:
            processed = series.rank(ascending=True)

        if isinstance(raw_data, dict):
            processed = processed.to_dict()  # 如果raw_data是字典类型，则将processed转换成字典型

        return processed  # 标准化后的序列，类型与输入相同

    def my_dropstyle(self, raw_data, cap, stk_indus, mode=None, regress_mode='OLS', cap_mode='normal'):
        """> my_dropstyle(raw_data, stk_indus=None)
            将输入剔除风格因子的影响。
            Inputs:
                raw_data: [dict/pd.Series/np.array] 输入待处理数据
                cap: [pd.Series] index为股票，value为市值
                stk_indus: [pd.Series] index为股票，value为所属行业
                mode:[int] 剔除风格因子（做回归）的模式（待补充）
                    0: y = β行业 + ε
                    1: y = β1行业 + β2市值 + ε
                    2: y = β1行业 + β2市值 + β3市值**2 +β4市值**3 +ε
                    3: y = β1行业 + β2市值 + β3行业 * 市值 +ε
                    4: y = β1行业 + β2市值 + β3市值**2 +β4市值**3 + β5行业 * 市值 + ε
            Returns:
                processed: 回归残差序列，类型与输入相同
            """
        mode = mode or self.dropstyle_mode
        if raw_data.name == 'LNCAP':
            return raw_data

        if isinstance(raw_data, dict):  # 如果raw_sec是dict类型
            series = pd.Series(raw_data)  # 将raw_sec转化为Series类型
        else:
            series = raw_data  # 如果raw_sec不是dict类型则保持其不变

        if cap_mode == 'log':
            cap = np.log(cap)
        else:
            pass

        try:  # 尝试dropstyle
            indus_fct_expos = pd.get_dummies(stk_indus)
            # 规范命名
            series.name = 'factor'
            cap.name = 'cap'
            full_df = pd.merge(series, indus_fct_expos, left_index=True, right_index=True)
            full_df = pd.merge(full_df, cap, how='left', left_index=True, right_index=True)
            full_df.dropna(subset=['factor', 'cap'], inplace=True)  # 避免有空值导致回归出问题
            full_df['cap'] = full_df['cap'].astype(np.int64)
            cap_weights = full_df['cap'] / full_df['cap'].sum()
            endog = full_df.loc[:, ['factor']].values  # endog = 类型是Series，索引是股票的wind代码，内容是只保留full_df的第一列
            if mode == 0:  # 只对行业中性化
                exog = full_df.iloc[:, ~full_df.columns.isin(
                    ['factor', 'cap'])].values  # exog = 类型是DataFrame，索引是股票的wind代码，内容是只保留full_df的第二列到最后一列
            elif mode == 1:  # 对行业和市值中性化
                exog = full_df.iloc[:, ~full_df.columns.isin(
                    ['factor'])].values  # exog = 类型是DataFrame，索引是股票的wind代码，内容是只保留full_df的第二列到最后一列
            elif mode == 2:  # 对行业和市值高阶多项式（三阶）中性化
                full_df['cap_2'] = full_df['cap'] ** 2  # 市值的平方
                full_df['cap_3'] = full_df['cap'] ** 3  # 市值的三次方
                exog = full_df.iloc[:, ~full_df.columns.isin(
                    ['factor'])].values  # exog = 类型是DataFrame，索引是股票的wind代码，内容是只保留full_df的第二列到最后一列
            elif mode == 3:  # 对行业、市值及其交互项做中性化
                industry_list = list(full_df.iloc[:, ~full_df.columns.isin(['factor', 'cap'])].columns)
                for icol in industry_list:
                    full_df[icol + '_cross_cap'] = full_df[icol] * full_df['cap']
                exog = full_df.iloc[:, ~full_df.columns.isin(
                    ['factor'])].values  # exog = 类型是DataFrame，索引是股票的wind代码，内容是只保留full_df的第二列到最后一列
            elif mode == 4:  # 对行业、市值
                industry_list = list(full_df.iloc[:, ~full_df.columns.isin(['factor', 'cap'])].columns)
                for icol in industry_list:
                    full_df[icol + '_cross_cap'] = full_df[icol] * full_df['cap']
                full_df['cap_2'] = full_df['cap'] ** 2  # 市值的平方
                full_df['cap_3'] = full_df['cap'] ** 3  # 市值的三次方
                exog = full_df.iloc[:, ~full_df.columns.isin(
                    ['factor'])].values  # exog = 类型是DataFrame，索引是股票的wind代码，内容是只保留full_df的第二列到最后一列
            else:
                raise ('mode参数输入错误，请检测输入格式（int），或重新输入参数（[0,4]）')
            exog = exog.astype(np.float64)
            endog = endog.astype(np.float64)
            if regress_mode == 'WLS':
                model = sm.WLS(endog, exog, weights=cap_weights)
            elif regress_mode == 'OLS':
                model = sm.OLS(endog, exog)
            result = model.fit()
            processed = pd.Series(result.resid, index=full_df.index)
        except ValueError as e:  # 如果ValueError，查看series情况
            raise e

        if isinstance(raw_data, dict):
            processed = processed.to_dict()# 如果raw_data是字典类型，则将processed转换成字典型

        return processed


class ProcessFactor:
    def __init__(self, factor_list, data_path=None, list_date_limit=90, start=None, end=None):
        self.start = start or '20200101'
        self.end = end or '20250701'
        self.data_path = data_path or fr'{FactorPath.datacenter_ssd_code}:\local_data\Data_Storage'
        self.factor_name = factor_list
        self.ticker_info = feather.read_dataframe(os.path.join(self.data_path, 'ticker_info.feather'))

        self.ticker_info['start_date'] = self.ticker_info['start_date'].apply(lambda x: (
                pd.to_datetime(x) + dt.timedelta(days=list_date_limit)).strftime('%Y%m%d'))
        self.ticker_info['end_date'] = self.ticker_info['end_date'].apply(lambda x: (
            pd.to_datetime(x)).strftime('%Y%m%d'))

    def load_data(self, ST=False):
        citic = feather.read_dataframe(os.path.join(self.data_path, 'CITIC_CODE.feather'))
        citic['citic_code'] = citic['CITIC_CODE'].str[:5]
        citic = citic[['TICKER','DATE','citic_code']]

        # citic['citic_code'] = citic['citic_code'].apply(lambda x: x[:6])
        cap_value = feather.read_dataframe(os.path.join(self.data_path, 'float_mv.feather'),
                                           columns=['TICKER', 'DATE', 'float_mv'])
        cap_value.columns = ['TICKER', 'DATE', 'mv']
        cap_value['mv'] = cap_value['mv']/10000
        cap_value = cap_value.pivot(index='DATE', values='mv', columns='TICKER').ffill().melt(ignore_index=False, value_name='mv')
        cap_value = cap_value.sort_values(by=['TICKER', 'DATE'])

        df_daily = feather.read_dataframe(os.path.join(self.data_path, 'daily.feather'),
                                          columns=['TICKER', 'DATE', 'close'])
        df_adj_factors = feather.read_dataframe(os.path.join(self.data_path, 'adj_factors.feather'))
        df_daily = pd.merge(df_daily,df_adj_factors,how='left',on=['TICKER','DATE'])
        df_daily['adj_close'] = df_daily['close']*df_daily['adj_factors']
        df_daily['adj_pre_close'] = df_daily.groupby('TICKER')['adj_close'].shift(1)
        df_daily = df_daily[['TICKER','DATE','adj_close','adj_pre_close']]
        df_daily = df_daily.dropna().reset_index(drop=True)

        df_daily['log_return'] = np.log(df_daily['adj_close'] / df_daily['adj_pre_close'])
        # date_max = df_daily.DATE.max()
        df_daily = df_daily.pivot(index='DATE', values='log_return', columns='TICKER').ffill()
        df_daily = np.exp(df_daily.rolling(120, min_periods=60).sum())
        df_daily = df_daily.sub(df_daily.mean(axis=1), axis='index')
        df_daily = df_daily.melt(ignore_index=False, value_name='excess_return').sort_values(by=['TICKER', 'DATE'])

        df_basic = pd.merge(cap_value, citic, on=['TICKER', 'DATE'], how='outer')
        if ST:
            df_st = feather.read_dataframe(os.path.join(self.data_path, 'st_all.feather'),
                                           columns=['TICKER', 'DATE', 'ST_status'])
            df_st = df_st.pivot(index='DATE', values='ST_status', columns='TICKER').ffill().melt(ignore_index=False, value_name='ST_status')
            df_st = df_st.sort_values(by=['TICKER', 'DATE'])
            df_basic = pd.merge(df_basic, df_st, on=['TICKER', 'DATE'], how='outer')
        else:
            pass

        df_basic = pd.merge(df_basic, df_daily, on=['TICKER', 'DATE'], how='outer')
        df_basic = df_basic[df_basic['TICKER'].str[-2:].isin(['SH','SZ'])]
        df_basic = df_basic[df_basic['DATE'] <= self.end]
        df_basic = df_basic[df_basic['DATE'] >= self.start].reset_index(drop=True)

        return df_basic

    def load_factor(self, df_basic, df_raw_data, ST=True):
        df_raw_data = df_raw_data[['TICKER', 'DATE'] + self.factor_name]
        df_raw_data = pd.merge(df_raw_data, self.ticker_info, on='TICKER', how='left')
        df_raw_data = df_raw_data[
            (df_raw_data['DATE'] >= df_raw_data['start_date']) & (df_raw_data['DATE'] <= df_raw_data['end_date'])]
        df_basic = df_basic[
            (df_basic['DATE'] >= df_raw_data['DATE'].min()) & (df_basic['DATE'] <= df_raw_data['DATE'].max())]
        df_raw_data = pd.merge(df_raw_data, df_basic, on=['TICKER', 'DATE'], how='left')

        if ST:
            df_raw_data = df_raw_data[df_raw_data['ST_status'] != 1]
            df_raw_data = df_raw_data.drop('ST_status', axis=1)
        else:
            pass

        df_raw_data.set_index('TICKER', inplace=True)
        # df_raw_data.rename(columns={'CITIC_CODE':'citic_code'},inplace=True)
        df_raw_data = df_raw_data.dropna(subset=['mv', 'citic_code'])
        df_raw_data = df_raw_data.drop(['start_date', 'end_date'], axis=1)
        return df_raw_data


    def __call__(self, df_raw_data, factor=None, fillna=True, win=True, stand=True, dropstyle=True,
                 win_type='NormDistDraw_zip',
                 dropstyle_mode=1, n_draw=1, pvalue=.05, stand_weight=None, fillna_type='median', stand_type='zscore',
                 cap_mode='normal', regress_mode='OLS', process_order=None, merge_cap=True):
        factor = factor or self.factor_name[0]
        if merge_cap:
            df_raw_data = self.load_data(df_raw_data)
        else:
            df_raw_data = df_raw_data
        df_columns = df_raw_data.columns.drop(self.factor_name).to_list() + [factor]
        df_raw_data = df_raw_data[df_columns]
        # df_raw = df_raw.dropna()
        sig_processor = SignalProcessor(fillna=fillna, win=win, stand=stand, dropstyle=dropstyle)
        tqdm.pandas(desc='Processing %s' % factor)

        df_processed = df_raw_data.groupby('DATE').progress_apply(
            lambda x: sig_processor(raw_data=x[factor], cap=x['mv'], stk_indus=x['citic_code'],
                                    cap_mode=cap_mode, regress_mode=regress_mode, process_order=process_order,
                                    dropstyle_mode=dropstyle_mode, stand_type=stand_type,
                                    n_draw=n_draw, pvalue=pvalue, stand_weight=stand_weight, fillna_type=fillna_type,
                                    win_type=win_type))

        df_processed = df_processed.reset_index().sort_values(by=['TICKER', 'DATE'])
        return df_processed[['TICKER','DATE',factor]].reset_index(drop=True)


if __name__ == '__main__':
    '''
    如下是一个中性化样例
    中性化分为几个步骤：读取文件，中性化处理，输出文件
    【读取文件】可以用read_file函数，或者自行进行更改
    【中性化处理】
    【输出文件】此处涉及一个逻辑，因为我们经常会在一个文件里存放多个因子，那么在输出时，
               我们可以选择一个因子保存一个processed文件，也可以该原始文件内的所有因子，
               继续保存在同一个processed文件里面。这个逻辑就涉及到了下面的【split_out】参数，
               以及out_put_name_dict和out_put_folder_dict的设置
    '''
    '''
    split_output: 是否分开输出文件内的多个因子
    factor_name_dict: 格式 {file:[factor_list]} 表示每个文件要process哪几个因子
                      可以为空，则默认process全部因子
    file_name_list: 文件列表
    
    【out_put_name_dict】 格式 file_name: output_file_name / 当split_out为False时，该dict不能为空
                          这个写明了，输出文件名，表示原始文件中全部因子中性化之后，保存的那一个文件的名字
    【out_put_folder_dict】 格式 factor_name: output_file_name / 当split_out为True时，该dict不能为空
                          这个写明了每个因子输出到哪个文件夹，输出名则默认为：因子名_processed_rg.feather
    
    data_path 输入基础数据文件路径
    factor_path 输入因子文件路径
    out_put_path 输出文件路径
    
    cap_mode='normal',  用于中性化回归的市值的量纲，normal 或 log
    regress_mode='OLS',  中性化回归的方式 OLS 或 WLS
    n_draw=1, 不用管
    stand_type='rank_gauss', 标准化方式 zscore 或 rank_gauss
    win_type='MAD',       去极值方式 MAD 或 QuantileDraw 或 NormDistDraw
    fillna_type='median_global',  fillna的方法 mean/median + '_global'/''
                                  当后面有'_global'时代表没有被填充的缺失值，最后要来一步全局填充。
    merge_cap=False, 不用管
    
    fillna=True 四个参数表示要不要做这一步
    win=True    process的过程就是按照这个从上到下的顺序
    stand=True
    dropstyle=True)
    '''

    '''
    如果外部调用，可以不做out_put的一系列设置
    '''

    split_output = False
    factor_path = r'F:\Factor_Storage\周楷贺\原始数据\拟使用量价因子'
    out_put_path = r'F:\Factor_Storage\周楷贺\中性化数据\拟使用量价因子'
    data_path = r'U:\local_data\Data_Storage'

    # file_name_list = ['REV_4_similarity.feather']
    file_name_list = os.listdir(factor_path)
    # 这表示要进行中性化的是保存在E:\vp_zkh\vp_factors\REV_4_similarity下的factor文件夹的REV_4_similarity.feather
    factor_name_dict = {}
    out_put_name_dict = {}  # 这个写了下一条语句就不会运行，如果是空的，则默认在原始文件后面加个processed输出
    out_put_name_dict = {x: x.replace('.feather', '_processed.feather') for x in file_name_list} if len(
        out_put_name_dict) == 0 else out_put_name_dict
    # 输出到factor这个文件夹内，文件名默认为 因子名_processed.feather

    for j in range(len(file_name_list)):
        file_name = file_name_list[j]
        try:
            # 读取文件
            df_original_data = read_file(factor_path, file_name)
            try:
                factor_name_list = factor_name_dict[file_name]
            except:
                factor_name_list = df_original_data.columns.drop(['TICKER', 'DATE']).to_list()

            # 生成中性化class的实例
            processor = ProcessFactor(data_path=data_path, factor_list=factor_name_list,start='20200101',end='20250905')
            # 加载数据，添加columns 行业、市值
            df_whole = processor.load_data(df_original_data)

            # 循环每一列进行中性化处理
            # for i in range(len(factor_name_list)):
            # stand_type 默认zscore
            factor_name = factor_name_list[0]
            df_processed = processor(df_raw_data=df_whole, factor=factor_name, cap_mode='normal', regress_mode='OLS',
                                     n_draw=1, stand_type='ZSCORE',
                                     win_type='MAD', fillna_type='median', merge_cap=False, fillna=True)

            # 输出文件!可自行更改
            out_put_name = out_put_name_dict[file_name]
            feather.write_dataframe(df_processed, os.path.join(out_put_path, out_put_name))
        except:
            print(file_name)

    os.makedirs(r'\\192.168.1.210\Factor_Storage\周楷贺\中性化完成标识\20250905',exist_ok=True)