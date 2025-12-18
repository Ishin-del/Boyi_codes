import datetime
import time

import pandas as pd
import copy
import numpy as np
import datetime as dt
import os
import feather
import openpyxl as op
import sys
from tqdm import tqdm
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator as mpl

import scipy.stats as sp
import statsmodels.api as sm
from statsmodels.stats.weightstats import DescrStatsW
from empyrical import (annual_return, sharpe_ratio)

sys.path.append(r'../')
# from config.data_path import FactorPath
from tool_tyx.data_path import FactorPath

from zkh_funcs import *


def del_new_ticker(x):
    x = x.sort_values(by=['DATE']).reset_index(drop=True)
    x['rank'] = [i + 1 for i in range(len(x))]
    x = x[x['rank'] > 20]
    x = x.drop('rank', axis=1)
    return x


plt.rcParams['font.sans-serif'] = ['FangSong']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')


class Factor_Test():
    def __init__(self, start=None, end=None, use_vwap_sig='vwap_1000_1030', date_shift=None, use_type=None,
                 special_condition=None):
        self.start = start or '20220101'
        self.end = end or '20250627'
        self.use_vwap_sig = use_vwap_sig or 'vwap_930_1000'
        self.date_shift = date_shift or 1
        self.use_type = use_type
        self.special_condition = special_condition

    def load_Data(self):
        self.calendar = pd.read_csv(FactorPath.calender_path, dtype={'trade_date': str})
        self.daily = feather.read_dataframe(os.path.join(FactorPath.data_storage, 'daily.feather'),
                                            columns=['TICKER', 'DATE'])

        # 去除新股和ST
        self.daily = self.daily.groupby('TICKER').apply(del_new_ticker).reset_index(drop=True)
        st = pd.read_csv(os.path.join(FactorPath.base_data_path, 'st.csv'), encoding='gbk')
        st = st.set_index('Unnamed: 0')
        st.index.name = 'DATE'
        st = pd.melt(st, ignore_index=False)
        st = st.reset_index(drop=False)
        st.columns = ['DATE', 'TICKER', 'ST']
        st['DATE'] = st['DATE'].apply(lambda x: ''.join(x.split('-')))
        st = st[st['TICKER'].str[-2:].isin(['SH', 'SZ'])]
        self.daily = pd.merge(self.daily, st, how='left', on=['TICKER', 'DATE'])
        self.daily = self.daily[self.daily['ST'] != '是']

        self.daily = self.daily[(self.daily['DATE'] <= self.end) & (self.daily['DATE'] >= self.start)]
        self.daily = self.daily.drop('ST', axis=1).reset_index(drop=True)

        # 导入复权系数和价格数据
        self.adj_factors = feather.read_dataframe(os.path.join(FactorPath.data_storage, 'adj_factors.feather'))
        self.adj_factors = pd.merge(self.adj_factors, self.daily, on=['TICKER', 'DATE'], how='right').reset_index(
            drop=True)

        if self.use_vwap_sig in ['open', 'close']:
            self.price = feather.read_dataframe(os.path.join(FactorPath.data_storage, f'daily.feather'),
                                                columns=['TICKER', 'DATE', self.use_vwap_sig])

        elif f'{self.use_vwap_sig}.feather' in os.listdir(FactorPath.data_storage):
            self.price = feather.read_dataframe(os.path.join(FactorPath.data_storage, f'{self.use_vwap_sig}.feather'))
        else:
            print('Price data not exist!')
            exit(0)

        if self.use_type == 'today':
            shift_date = 0
        elif self.use_type == 'tomorrow':
            shift_date = 1
        elif 'D' in self.use_type.upper():
            shift_date = int(self.use_type.replace('D', ''))
        else:
            raise ValueError

        if not isinstance(self.special_condition, str):
            raise ValueError('ticker pool is empty!')

        pools = feather.read_dataframe(self.special_condition,columns=['TICKER','DATE'])
        pools = pools[pools['DATE'] <= self.end]
        pools = pools[pools['DATE'] >= self.start]

        self.daily = pd.merge(pools,self.daily,on=['TICKER','DATE'],how='left')

        self.price = pd.merge(self.price, self.daily, on=['TICKER', 'DATE'], how='right').reset_index(drop=True)
        self.price = self.price.dropna().reset_index(drop=True)
        price_name = list(np.setdiff1d(self.price.columns, ['TICKER', 'DATE']))[0]
        self.price = self.price.rename(columns={price_name: 'price'})
        self.price = pd.merge(self.price, self.adj_factors, how='left', on=['TICKER', 'DATE'])
        self.price['price'] = self.price['price'] * self.price['adj_factors']
        self.price = self.price.sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
        self.price['Return'] = self.price.groupby('TICKER')['price'].shift(-shift_date - self.date_shift).reset_index(
            drop=True) / self.price.groupby('TICKER')['price'].shift(-shift_date).reset_index(drop=True)

        self.price['Return'] = (self.price['Return'] - 1) * 100
        self.price = self.price[['TICKER', 'DATE', 'Return']]

    def load_factor_data(self, factor_path):
        """
        factor_path 因子路径
        use_type ['today','tomorrow'] today 专为集合竞价因子 
        special_condition 特殊股票池，考虑链接方向
        """
        try:
            factor = feather.read_dataframe(factor_path)
        except Exception as e:
            print(factor_path, e)

        if factor.shape[1] != 3:
            raise ValueError

        if 'TICKER' not in factor.columns or 'DATE' not in factor.columns:
            raise ValueError
        price_Data = self.price
        factor = factor[(factor['DATE'] <= self.end) & (factor['DATE'] >= self.start)].reset_index(drop=True)

        all_data = pd.merge(factor, price_Data, how='right', on=['TICKER', 'DATE']).reset_index(drop=True)
        factor_name = list(np.setdiff1d(all_data.columns, ['TICKER', 'DATE', 'Return']))[0]
        return all_data, factor_name

    def cal_factors_validation(self, x, factor_name):
        d = x['DATE'].values[0]
        if 'DATE' in x.columns:
            x = x.drop(['DATE'], axis=1)
        if 'TICKER' in x.columns:
            x = x.drop(['TICKER'], axis=1)
        res = x.corr(method='spearman')
        res = res[[factor_name]]
        res = res.T
        res = res.drop(factor_name, axis=1)
        res = res.reset_index(drop=True)
        res['DATE'] = d
        return res

    def cal_ic(self, x, y, method='normal', weight=None):
        """【功能性函数】计算IC
        Parameters
        ----------
        x: 一般是factor序列
        y: 一般是收益序列

        weight:IC权重, None or Series, Default None
            默认为None，等权计算IC；如果采用加权IC计算方法，则输入权重序列
        Returns
        -------
            IC值: float
        """
        if method == 'normal':
            if weight is not None:
                weight_series = DescrStatsW(np.array([x, y]).T, weights=weight)
                pcc = weight_series.corrcoef[0, 1]
            else:
                n = len(x.dropna())
                xba = np.mean(x)
                yba = np.mean(y)
                xstd = np.std(x)
                ystd = np.std(y)
                pcc = np.sum((x - xba) * (y - yba)) / (n * xstd * ystd)
            return pcc
        elif method == 'rank':
            pcc = sp.spearmanr(x, y).correlation
            return pcc

    def calc_Factors_metrics(self, factor_data, factor_name, bin_num, plot_save_path):
        plot_save_path=os.path.join(plot_save_path,factor_name)
        os.makedirs(plot_save_path,exist_ok=True)
        # region 计算因子指标
        factor_data = factor_data.dropna(subset=['Return']).reset_index(drop=True)
        ic_results = factor_data.groupby('DATE').apply(self.cal_factors_validation,
                                                       factor_name=factor_name).reset_index(drop=True)
        bins = {}
        for key, series in tqdm(factor_data.set_index('TICKER').groupby('DATE')[factor_name],
                                desc=f'{factor_name} 分层'):
            # print(key)
            if key in ['20240206', '20240207', '20240208', '20240926', '20240927','20240930', '20241008']:
                continue
            # print(key)
            # print(series)
            try:
                bins[key] = pd.qcut(series, q=bin_num, labels=range(1, bin_num + 1))
            except:
                series.drop_duplicates(inplace=True)
                bins[key] = pd.qcut(series, q=bin_num, labels=range(1, bin_num + 1))

        df_bins = pd.melt(pd.DataFrame(bins), ignore_index=False, value_name='bins',
                          var_name='DATE').reset_index(drop=False)
        factor_data.reset_index(drop=True, inplace=True)
        factor_data = pd.merge(factor_data, df_bins, on=['TICKER', 'DATE'], how='left')
        factor_data['exchange_date'] = factor_data['DATE']
        factor_data_split = factor_data.groupby('bins').agg({'Return': ['mean', 'std']}).reset_index(drop=False)
        factor_data_split.columns = ['bins', 'Return_mean', 'Return_std']
        factor_data_split['IR_split_by_bins'] = factor_data_split['Return_mean'] / factor_data_split['Return_std']
        split_by_date = factor_data.groupby(['DATE', 'bins']).agg({'Return': 'mean'}).reset_index(drop=False)
        split_by_date['Return'] = split_by_date['Return'] / 100 + 1
        split_by_date['Return_cumprod'] = split_by_date.groupby('bins')['Return'].cumprod().reset_index(drop=True)
        pvt_split_by_date = pd.pivot_table(split_by_date, columns=['bins'], index=['DATE'], values='Return')

        mean_ic = np.mean(ic_results['Return'])
        std_ic = np.std(ic_results['Return'], ddof=1)
        n = len(ic_results['Return'])
        t_value_ic = mean_ic / (std_ic / np.sqrt(n))

        if mean_ic > 0:
            pvt_split_by_date['long_short_Return'] = pvt_split_by_date[bin_num] / pvt_split_by_date[1]
            pvt_split_by_date['long_short_Return'] = abs(pvt_split_by_date['long_short_Return'])
            long_side = factor_data[factor_data['bins'] == bin_num][['Return', factor_name, 'DATE']]
        else:
            pvt_split_by_date['long_short_Return'] = pvt_split_by_date[1] / pvt_split_by_date[bin_num]
            pvt_split_by_date['long_short_Return'] = abs(pvt_split_by_date['long_short_Return'])
            long_side = factor_data[factor_data['bins'] == 1][['Return', factor_name, 'DATE']]

        longside_metrics = long_side.groupby('DATE').apply(self.cal_factors_validation,
                                                           factor_name=factor_name).reset_index(drop=True)
        longside_metrics_ic = longside_metrics['Return'].mean()
        longside_metrics_ic_std = longside_metrics['Return'].std()
        longside_metrics_ic_ir = longside_metrics_ic / longside_metrics_ic_std

        pvt_split_by_date_cumprod = pvt_split_by_date.cumprod()

        ic_results = ic_results.rename(columns={'Return': 'IC'})
        ic_results['accumulate_IC'] = ic_results['IC'].cumsum()
        ic_results = ic_results.set_index('DATE')
        # endregion

        # region 画图
        df_factor = factor_data.dropna(subset=[factor_name])
        pct_all_dict = {}

        for i in range(1, bin_num + 1):
            pct_all_dict[i] = {}

        for trade_date, df_date in tqdm(df_factor.groupby('DATE'), desc='计算各层的平均收益'):
            for bin in range(1, bin_num + 1):
                df_temp = df_date[df_date['bins'] == bin]
                # df_temp.drop_duplicates(subset = ['TICKER'], inplace = True)
                return_data = df_temp['Return']

                if return_data.isna().min():
                    pass
                else:
                    pct_all_dict[bin][trade_date] = return_data.mean() + 1

        df_exchange_return = pd.DataFrame(pct_all_dict)
        (df_exchange_return.mean() - 1).plot.bar(figsize=(16, 9),
                                                 title=f'{factor_name}_average_exchange_return_bins')
        plt.ylabel('收益率', fontsize=16)
        plt.xlabel('箱位（箱位值越高因子值越大）', fontsize=16)

        if plot_save_path is not None:
            plt.savefig(os.path.join(plot_save_path,
                                     '%s_各层每个调仓周期内的平均收益_%s.png' % (factor_name, self.date_shift)))

        plt.close()

        # IC图
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
        plt.title(f'{factor_name} 因子的Rank_IC')
        l1 = ax1.bar(x=ic_results['IC'].index,
                     height=ic_results['IC'],
                     label='IC', color='0.8')
        l2 = ax1.axhline(y=ic_results['IC'].mean(), c="red", linestyle='--')
        ax1.set_ylabel('因子Rank_IC值', fontsize=18)
        ax1.legend([l1, l2], ['Rank_IC值(左轴)', 'Rank_IC值均值(左轴)'], loc='upper left')
        # 坐标轴（右轴）
        ax2 = ax1.twinx()
        l3 = ax2.plot(ic_results['accumulate_IC'], label='累计Rank_IC', color='b')
        ax2.set_ylabel('累计Rank_IC值', fontsize=18)
        ax2.legend(l3, ('累计Rank_IC(右轴)',), loc='upper right')
        # 旋转横轴刻度字体
        plt.xticks(rotation=45)
        # 设置横坐标间隔
        x_major_locator = mpl(np.round(len(ic_results) / 8))
        ax1.xaxis.set_major_locator(x_major_locator)

        if plot_save_path is not None:
            fig.savefig(
                os.path.join(plot_save_path, '%s_Rank_IC_performance_%s.png' % (factor_name, self.date_shift)))
        plt.close()

        # IR
        bins_df_IC = {}
        bins_df_Rank_IC = {}
        print('计算IR并绘图')
        for bin, df_temp1 in tqdm(factor_data.groupby('bins'), desc=f'{factor_name}因子计算IC'):
            IC_series = pd.Series()
            RANK_IC_series = pd.Series()
            for exchange_date, igroup in df_temp1.groupby('DATE'):
                try:
                    factor_data_this = igroup[factor_name][igroup.index.unique()]
                    if factor_data_this.isna().sum() == len(factor_data_this):
                        continue
                    return_data_this = igroup['Return'][factor_data_this.index]
                    # if bins_weight is not None:
                    #     IC_series = IC_series.append(pd.Series(self.cal_ic(factor_data, return_data,
                    #                                                        weight=df_temp['ic_weight']),
                    #                                            index=[exchange_date]))  # 计算pearson correlation(IC)
                    # else:
                    IC_series = IC_series._append(
                        pd.Series(self.cal_ic(factor_data_this, return_data_this, weight=None),
                                  index=[exchange_date]))  # 计算pearson correlation(IC)
                    RANK_IC_series = RANK_IC_series._append(
                        pd.Series(sp.spearmanr(factor_data_this, return_data_this).correlation,
                                  index=[exchange_date]))
                except:
                    pass

            IC_series.name = factor_name + '_IC'
            RANK_IC_series.name = factor_name + '_Rank_IC'
            bins_df_IC[bin] = IC_series
            bins_df_Rank_IC[bin] = RANK_IC_series

        df_bins_df_IC = pd.DataFrame(bins_df_IC)
        df_bins_df_Rank_IC = pd.DataFrame(bins_df_Rank_IC)
        ax = (df_bins_df_IC.mean() / df_bins_df_IC.std()).plot.bar(figsize=(16, 9),
                                                                   title=f'{factor_name}_分箱IR')
        ax.set_xlabel('箱位序号', fontsize=16)
        ax.set_ylabel('IR', fontsize=16)
        if plot_save_path is not None:
            plt.savefig(os.path.join(plot_save_path, '%s_IR_performance_%s.png' % (factor_name, self.date_shift)))
        plt.close()

        ax = (df_bins_df_Rank_IC.mean() / df_bins_df_Rank_IC.std()).plot.bar(figsize=(16, 9),
                                                                             title=f'{factor_name}_分箱Rank_ICIR')
        ax.set_xlabel('箱位序号', fontsize=16)
        ax.set_ylabel('IR', fontsize=16)
        if plot_save_path is not None:
            plt.savefig(
                os.path.join(plot_save_path, '%s_Rank_ICIR_performance_%s.png' % (factor_name, self.date_shift)))
        plt.close()

        # 多空图
        # pvt_split_by_date
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
        plt.title(f'{factor_name}_多空组合累计收益曲线', fontsize=18)
        sr = sharpe_ratio(pvt_split_by_date['long_short_Return'] - 1)
        ar = annual_return(pvt_split_by_date['long_short_Return'] - 1)
        if mean_ic > 0:
            sr_long = sharpe_ratio(pvt_split_by_date[bin_num] - 1)
        else:
            sr_long = sharpe_ratio(pvt_split_by_date[1] - 1)

        plt.suptitle('sharpe_ratio = %.2f, annual_return = %.2f' % (sr, ar))
        #
        l1 = ax1.plot(pvt_split_by_date_cumprod[bin_num], label='箱位%d' % bin_num, color='b', linewidth=1,
                      linestyle='--')
        l2 = ax1.plot(pvt_split_by_date_cumprod[1], label='箱位1', color='r', linewidth=1, linestyle='--')
        ax1.set_ylabel('单箱位累计收益', fontsize=15)
        ax1.legend(loc='upper left')
        # # 坐标轴（右轴）
        ax2 = ax1.twinx()
        l3 = ax2.plot(pvt_split_by_date_cumprod['long_short_Return'], label='多空组合', color='y', linewidth=3)
        ax2.set_ylabel('多空组合累计收益', fontsize=15)
        ax2.legend(loc='upper right')
        # 旋转横轴刻度字体
        plt.xticks(rotation=45)
        # 设置横坐标间隔
        x_major_locator = mpl(np.round(len(pvt_split_by_date_cumprod) / 8))
        ax1.xaxis.set_major_locator(x_major_locator)

        if plot_save_path is not None:
            plt.savefig(os.path.join(plot_save_path,
                                     '%s_Long_Short_Cummulative_Return_%s.png' % (factor_name, self.date_shift)))
        plt.close()

        # 分层图
        fig, ax1 = plt.subplots(1, 1, figsize=(9, 6))
        plt.title(f'{factor_name}_分组累计收益曲线', fontsize=18)

        color_list = [
            '#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
            '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'
        ]

        # 你这里bin_num是整数，需要转换成范围
        for i in range(bin_num):
            ax1.plot(
                pvt_split_by_date_cumprod[i + 1],
                label=f'箱位{i + 1}',
                color=color_list[i],
                linewidth=1,
                linestyle='--'
            )

        ax1.set_ylabel('分箱位累计收益', fontsize=15)
        plt.xticks(rotation=45)

        # 设置横坐标刻度
        from matplotlib.ticker import MultipleLocator
        x_major_locator = MultipleLocator(np.round(len(pvt_split_by_date_cumprod) / 8))
        ax1.xaxis.set_major_locator(x_major_locator)

        # 添加图例：loc 可以是 'upper left'、'upper right' 等
        ax1.legend(loc='upper left', fontsize=10, frameon=False)

        # 保存图像
        if plot_save_path is not None:
            plt.savefig(os.path.join(
                plot_save_path,
                f'{factor_name}_Bins_Cummulative_Return_{self.date_shift}.png'
            ))

        plt.close()

        results = pd.DataFrame()
        results['factor_name'] = [factor_name]
        results['RANK_IC'] = mean_ic
        results['T值'] = t_value_ic
        results['多空夏普'] = sr
        results['多头夏普'] = sr_long
        results['多头RANK_IC'] = longside_metrics_ic
        results['多头ICIR'] = longside_metrics_ic_ir
        results['ICIR'] = mean_ic / std_ic
        results['factor_path'] = factor_path
        results.to_excel(os.path.join(plot_save_path, f'{factor_name}.xlsx'), index=False)
        # todo:
        # results.to_excel(os.path.join(r'\\Desktop-79nue61\因子测试结果\田逸心\因子中性化回测', f'{factor_name}.xlsx'), index=False)
        # endregion

if __name__ == '__main__':

    warnings.filterwarnings(action='ignore')
    # 设置参数
    info_list = get_Multi_info('请输入测试参数',
                               [('单选', '因子是否当天生成当天使用', ['是', '否']),
                                ('单选', '因子开发人', ['周楷贺', '吴文博', '周子逸', '田逸心', '高嘉伟', '实习生']),
                                ('单选', '测试时段', ['open', 'close', 'vwap_930_1000', 'vwap_930_1030', 'vwap_930_1130'
                                    , 'vwap_1000_1030', 'vwap_1300_1500', 'vwap_1400_1500', 'vwap_1430_1500']),
                                ('单选', '因子测试股票池', ['wind微盘', '全市场', '中证2000','深圳市场','上海市场']),
                                ('单选', '因子分箱数', ['5', '10'])])

    param_list = getMultiInput('请输入参数',
                               ['因子测试起始日期(默认20220104)', '因子测试结束日期(默认20250701)', '因子文件路径',
                                '因子目标未来天数'])
    today_factor, researcher, price_anchor, use_market, bin_split = info_list
    start_Date, den_Date, factors_path, aim_Date = param_list

    # 转换参数类型
    use_type = 'tomorrow' if today_factor == '否' else 'today'

    if use_market == '中证2000':
        special_condition = r"\\192.168.1.101\local_data\Data_Storage\index_932000.feather"
    elif use_market == 'wind微盘':
        special_condition = r"\\192.168.1.101\local_data\Data_Storage\wind_micro_index_stock\min_total_mv_400_stock_start_2019.feather"
    elif use_market == '深圳市场':
        # special_condition = r"\\192.168.1.101\local_data\Data_Storage\daily_sz.feather"
        special_condition = r"Z:\local_data\Data_Storage\daily_sz.feather"
    elif use_market == '上海市场':
        special_condition = r"\\192.168.1.101\local_data\Data_Storage\daily_sh.feather"
    else:
        # special_condition = r"\\192.168.1.101\local_data\Data_Storage\daily.feather"
        special_condition = r"Z:\local_data\Data_Storage\daily.feather"

    aim_Date = int(aim_Date)
    bin_split = int(bin_split)
    start_date = start_Date or '20220104'
    end_date = den_Date or '20250822'

    obj = Factor_Test(use_type=use_type, start=start_date, end=end_date, date_shift=aim_Date, use_vwap_sig=price_anchor,
                      special_condition=special_condition)
    obj.load_Data()
    today = datetime.datetime.today().strftime('%Y%m%d')
    ls = os.listdir(factors_path)
    ls = [i for i in ls if i.endswith('.feather')]
    # sig = os.path.exists(r'\\192.168.1.210\Factor_Storage\周楷贺\中性化完成标识\20250817')
    # while not sig:
    #     sig = os.path.exists(r'\\192.168.1.210\Factor_Storage\周楷贺\中性化完成标识\20250817')
    #     time.sleep(600)

    for factor_path in ls:
        try:
            factor, name = obj.load_factor_data(
                factor_path=os.path.join(factors_path, factor_path))
            # plt_save_path = rf'\\Desktop-79nue61\因子测试结果\{researcher}\{today}_{start_date}_{end_date}_{aim_Date}\{factor_path[:-8]}'
            # plt_save_path = rf'\\192.168.1.210\因子测试结果\{researcher}\{today}_{start_date}_{end_date}_{aim_Date}\{factor_path[:-8]}'
            plt_save_path = rf'C:\Users\admin\Desktop\回测'
            os.makedirs(plt_save_path, exist_ok=True)
            # todo:
            # os.makedirs(r'\\Desktop-79nue61\因子测试结果\田逸心\因子中性化回测', exist_ok=True)
            obj.calc_Factors_metrics(factor, name, bin_num=bin_split, plot_save_path=plt_save_path)
        except Exception as e:
            print(e)
