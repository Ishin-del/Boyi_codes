import warnings

import pandas as pd
import copy
import numpy as np
import datetime as dt
import os
import feather
import openpyxl as op
import pyecharts.options as opts
from pyecharts.charts import Line
from empyrical import (annual_return, max_drawdown, sharpe_ratio, calmar_ratio)
import quantstats as qs
import tkinter as tk
from tkinter import ttk
import sys

sys.path.append('../')
# from config.data_path import FactorPath


def getMultiInput(title, message, row_count=1):
    r = []

    def return_callback():
        for index, content in enumerate(message):
            str11 = v[index].get()
            r.append(str11)
        window.destroy()

    window = tk.Tk(className=title)
    window.wm_attributes('-topmost', 1)
    rows = int(len(message) // row_count)
    width = 700
    height = 700 + np.ceil(max(0, (rows - 10)) // 5) * 200
    screenwidth = window.winfo_screenwidth()
    screenheight = window.winfo_screenheight()
    window.geometry('%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))

    frame1 = tk.Frame(window)
    frame1.pack(pady=10, padx=10, expand=True)

    v = []
    for index, content in enumerate(message):
        lable = tk.Label(frame1, height=2)
        lable.grid(row=index * 2 + 1, column=0, sticky='w')
        lable['text'] = content

        entry = tk.Entry(frame1)
        entry.grid(row=index * 2 + 2, column=0, sticky='w')
        entry.focus_set()
        v.append(entry)

    ttk.Button(frame1, text="确认", command=return_callback).grid(row=len(message) * 2 + 2, column=0)
    window.mainloop()

    return r


def getSelection(title, options, row_count=4):
    # 点击勾选框触发
    # 反选
    def unselectall():
        for index, item in enumerate(options):
            v[index].set('')

    # 全选
    def selectall():
        for index, item in enumerate(options):
            v[index].set(item)

    # 获取选择项
    def showselect():
        s = [i.get() for i in v if i.get()]
        print('\033[1;33m{}{}\033[0m'.format("选择的是:", s))
        window.destroy()

    window = tk.Tk(className=title)
    window.wm_attributes('-topmost', 1)
    rows = np.ceil(len(options) // row_count)
    width = 700
    height = 500 + np.ceil(max(0, (rows - 10)) // 5) * 200
    screenwidth = window.winfo_screenwidth()
    screenheight = window.winfo_screenheight()
    window.geometry('%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))

    frame1 = tk.Frame(window)
    frame1.pack(pady=10, padx=10, expand=True)
    # frame1.grid(row=0, column=0)

    # 全选反选
    opt = tk.IntVar()
    ttk.Radiobutton(frame1, text='全选', variable=opt, value=1, command=selectall).grid(row=0, column=0, sticky='w')
    ttk.Radiobutton(frame1, text='反选', variable=opt, value=0, command=unselectall).grid(row=0, column=1, sticky='w')

    v = []
    # 设置勾选框，每四个换行
    for index, opt in enumerate(options):
        v.append(tk.StringVar())
        ttk.Checkbutton(frame1, text=opt, variable=v[-1], onvalue=opt, offvalue='').grid(row=index // row_count + 1,
                                                                                         column=index % row_count,
                                                                                         sticky='w')
    ttk.Button(frame1, text="确认", command=showselect).grid(row=index // row_count + 2, column=0)

    window.mainloop()
    s = [i.get() for i in v if i.get()]
    return s


def getSingalSelection(title, options):
    # 获取选择项
    def showselect():
        print('\033[1;33m{}{}\033[0m'.format("选择的是:", v.get()))
        window.destroy()

    window = tk.Tk(className=title)
    window.wm_attributes('-topmost', 1)
    width = 500
    height = 500
    screenwidth = window.winfo_screenwidth()
    screenheight = window.winfo_screenheight()
    window.geometry('%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))

    frame1 = tk.Frame(window)
    frame1.pack(pady=10, padx=10, expand=True)

    lable = tk.Label(frame1, height=2)
    lable.grid(row=0, column=0, sticky='w')
    lable['text'] = title

    v = tk.StringVar()
    v.set('Wrong')
    # 设置勾选框，每四个换行
    for index, opt in enumerate(options):
        ttk.Radiobutton(frame1, text=opt, value=opt, variable=v).grid(row=index // 4 + 1, column=index % 4, sticky='w')

    ttk.Button(frame1, text="确认", command=showselect).grid(row=index // 4 + 2, column=0)
    window.mainloop()
    s = v.get()
    return s


def get_Multi_info(title, info_list):
    r = []

    def return_callback():
        for j in range(len(info_list)):
            v_j = value_list[j]
            if isinstance(v_j, list):
                str11 = [s.get() for s in v_j]
            else:
                str11 = v_j.get()
            r.append(str11)
        window.destroy()

    window = tk.Tk(className=title)
    window.wm_attributes('-topmost', 1)
    rows = len(info_list)
    width = 700
    height = 700 + np.ceil(max(0, (rows - 4)) // 4) * 200
    screenwidth = window.winfo_screenwidth()
    screenheight = window.winfo_screenheight()
    window.geometry('%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))

    frame1 = tk.Frame(window)
    frame1.pack(pady=10, padx=10, expand=True)

    value_list = []
    index = 0
    for info in info_list:
        type1, name, content = info
        lable = tk.Label(frame1, height=2)
        lable.grid(row=index, column=0)
        lable['text'] = name
        index += 1

        if type1.lower() == '单选':
            v = tk.StringVar()
            v.set(content[0])
            for i, opt in enumerate(content):
                ttk.Radiobutton(frame1, text=opt, value=opt, variable=v).grid(row=i // 4 + index, column=i % 4,
                                                                              sticky='w')
            index = i // 4 + index + 1
            value_list.append(v)
        elif type1.lower() == 'input':
            entry = tk.Entry(frame1)
            entry.grid(row=index, column=0)
            entry.focus_set()
            value_list.append(entry)
            index = index + 1
        elif type1.lower() == '多选':
            v = []
            # 设置勾选框，每四个换行
            for i, opt in enumerate(content):
                v.append(tk.StringVar())
                ttk.Checkbutton(frame1, text=opt, variable=v[-1], onvalue=opt, offvalue='').grid(
                    row=i // 4 + index, column=i % 4, sticky='w')
            index = i // 4 + index + 1
            value_list.append(v)

    ttk.Button(frame1, text="确认", command=return_callback).grid(row=index + 2, column=0)
    window.mainloop()

    return r


def draw_pic(date_axis, hedge_return_daily, index_axis, port_axis, hedge_axis, pic_folder, pic_name, **kwargs):
    '''
    :param x_axis: date list
    :param index_axis:
    :param port_axis:
    :param hedge_axis:
    :param file_name:
    :return:
    '''

    daily_ret = hedge_return_daily - 1
    index_axis = index_axis / index_axis.iloc[0]
    port_axis = port_axis / port_axis.iloc[0]
    hedge_axis = hedge_axis / hedge_axis.iloc[0]

    ar = annual_return(daily_ret)
    md = max_drawdown(daily_ret)

    if 'date_drop' in kwargs.keys():
        date_drop = kwargs['date_drop']
        daily_ret = daily_ret[~daily_ret.index.isin(date_drop)]
    else:
        pass
    sr = sharpe_ratio(daily_ret)
    cm = calmar_ratio(daily_ret)

    picture = Line(init_opts=opts.InitOpts(width='1920px', height='900px',
                                           theme='white', page_title='page', bg_color='rgba(255,250,205,0.2)', ))
    subtitle = 'sharpe_ratio = %.2f, annual_return = %.2f, max_drawdown = %.4f, carma_ratio = %.2f' % (
        sr, ar, md, cm)
    if 'turnover' in kwargs.keys():
        subtitle = subtitle + ' turnover_rate = %.2f' % kwargs['turnover']
    picture.set_global_opts(
        title_opts=opts.TitleOpts(title='P&L',
                                  subtitle=subtitle,
                                  pos_left="center", pos_top="top"),
        tooltip_opts=opts.TooltipOpts(trigger='axis'),  # x对应值提示
        toolbox_opts=opts.ToolboxOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_='category', boundary_gap=False),
        yaxis_opts=opts.AxisOpts(min_='dataMin'),
        datazoom_opts=opts.DataZoomOpts(type_="inside", range_start=0, range_end=100),  # 放大缩小
        legend_opts=opts.LegendOpts(pos_left="left"),
    )
    picture.add_xaxis(xaxis_data=list(date_axis))
    picture.add_yaxis(
        series_name='Portfolio return after hedging.',
        y_axis=list(hedge_axis),
        is_connect_nones=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    picture.add_yaxis(
        series_name='Index return.',
        y_axis=list(index_axis),
        is_connect_nones=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    picture.add_yaxis(
        series_name='Portfolio return.',
        y_axis=list(port_axis),
        is_connect_nones=True,
        label_opts=opts.LabelOpts(is_show=False),
    )

    pic_name = 'BackTest_Pic_{0}.html'.format(pic_name)
    if len(pic_name) > 50:
        pic_name = pic_name
    pic_folder = pic_folder or 'E:/QuantWork/XCSX/Result_Pic/'

    daily_ret_qs = daily_ret.copy()
    daily_ret_qs.index = pd.to_datetime(daily_ret_qs.index)
    qs.reports.html(daily_ret_qs, output=os.path.join(pic_folder,'qs_report.html'))
    #
    # try:
    #     picture.render(os.path.join(pic_folder, pic_name))
    # except FileNotFoundError:
    replace_name = dt.datetime.now().strftime('%Y%m%d_%H%M%S') + '.html'
    picture.render(os.path.join(pic_folder, replace_name))
    with open(os.path.join(pic_folder, '名字替换txt.txt'), 'a+') as f:
        f.write('\n')
        f.write(f'{replace_name}\n   ：{pic_name}')
    f.close()
        # replace_name = replace_name

    # qt.reports.html(...)


class back_test:

    def __init__(self, start=None, end=None, test_goods=[], special_goods=None):
        self.start = start or '20230104'
        self.end = end or '20250829'
        self.data_path = r'\\192.168.1.101\local_data\Data_Storage'

        self.trade_cal = pd.read_csv(os.path.join(self.data_path, 'calendar.csv'), dtype={'trade_date': str})
        self.trade_cal['DATE'] = self.trade_cal['trade_date']
        self.ratio_result = pd.DataFrame()  # DataFrame used to save the results

        test_goods = [x for x in test_goods if len(x) > 0]
        if len(test_goods) == 0:
            self.test_goods = ['股票']
        else:
            self.test_goods = test_goods

    def set_params(self, args):
        '''
        :param args: a dict. Keys are the name of parameters, and values are the value of these params.
        '''

        keys = args.keys()
        if 'buy_tax' in keys:
            self.buy_tax = args['buy_tax']
        if 'sell_tax' in keys:
            self.sell_tax = args['sell_tax']
        if 'buy_comm' in keys:
            self.buy_commission = args['buy_comm']
        if 'sell_comm' in keys:
            self.sell_comm = args['sell_comm']
        if 'start_cash' in keys:
            self.start_cash = args['start_cash']

    def set_trade_params(self, args):
        '''
        设置一些交易时的特定参数
        '''
        keys = args.keys()
        if 'trade_time' in keys:
            self.trade_time = copy.deepcopy(args['trade_time'])
        if 'consider_ST' in keys:
            self.ST = copy.deepcopy(args['consider_ST'])
        if 'proportion' in keys:
            self.proportion = copy.deepcopy(args['proportion'])
        if 'price_type' in keys:
            # use which price to calculate the return(high,low,open,close)
            if type(args['price_type']) != list:
                self.price_type = [copy.deepcopy(args['price_type'])]
            else:
                self.price_type = copy.deepcopy(args['price_type'])
        else:
            self.price_type = ['open']
        self.price_type.append('st')
        if 'trade_freq' in keys:
            self.trade_freq = copy.deepcopy(args['trade_freq'])

    def set_pool(self, ticker_list=None):
        try:
            self.ticker_pool = list(self.portfolio['TICKER'].unique())
        except:
            self.ticker_pool = ticker_list
        self.ticker_pool.append(self.hedge_index)

    def set_hedge(self, hedge_index):
        '''
        设置用于对冲的股指
        '''
        self.hedge_index = hedge_index

    def get_trade_date(self, date_list=None):
        if date_list:
            self.trade_date = pd.DataFrame({'DATE': date_list})
        else:
            self.trade_date = self.trade_cal[['DATE']]
            self.trade_date = self.trade_date[
                (self.trade_date['DATE'] >= self.start) & (self.trade_date['DATE'] <= self.end)]

    def load_portfolio(self, file_name, folder=None, freq='M', select_date=False, trade_type='next_day',
                       del_warning=False, black_list=None):
        warnings.filterwarnings('ignore')
        '''
        无论Portfolio的日期对应的是哪一天 当我们最终将其赋予self.portfolio时 DATE都应该表示调仓当天
        '''
        file_folder = folder
        file_path = os.path.join(file_folder, file_name)
        file_type = file_name.split('.')[-1]
        self.file_name = file_name
        # 0. 黑名单股票
        if black_list is None:
            black_tickers = []
        else:
            bl_file_type = black_list.split('.')[-1]
            if bl_file_type == 'feather':
                black_tickers = feather.read_dataframe(black_list)
            elif bl_file_type == 'csv':
                try:
                    black_tickers = pd.read_csv(black_list, encoding='utf8 ', dtype={'DATE': str})
                except:
                    black_tickers = pd.read_csv(black_list, encoding='gb2312', dtype={'DATE': str})
            elif bl_file_type == 'xlsx':
                black_tickers = pd.read_excel(black_list, dtype={'DATE': str})
            else:
                raise Exception('黑名单文件格式有误，请检查')
            black_tickers = black_tickers.rename(columns={'DATE': 'black_date'}).drop('OPT_WEIGHT', axis=1)

        # 1. 导入持仓文件
        if file_type == 'feather':
            self.portfolio = feather.read_dataframe(file_path)
        elif file_type == 'csv':
            try:
                self.portfolio = pd.read_csv(file_path, encoding='utf8 ', dtype={'DATE': str})
            except:
                self.portfolio = pd.read_csv(file_path, encoding='gb2312', dtype={'DATE': str})
        elif file_type == 'xlsx':
            self.portfolio = pd.read_excel(file_path, dtype={'DATE': str})
        else:
            raise Exception('Portfolio文件格式有误，请检查')

        weight_col = [x for x in self.portfolio.columns if '权重' in x or 'weight' in x.lower()]
        assert len(weight_col) == 1, '权重列名称不对'

        self.portfolio = self.portfolio.rename(
            columns={'调整日期': 'DATE', '证券代码': 'TICKER', weight_col[0]: 'OPT_WEIGHT'})
        self.portfolio = self.portfolio.rename(
            columns={'date': 'DATE', 'ticker': 'TICKER', weight_col[0]: 'OPT_WEIGHT'})

        self.portfolio = self.portfolio[['TICKER', 'DATE', 'OPT_WEIGHT']].copy()

        # 2. 调整权重，截面和为1
        weight_daily = self.portfolio.groupby('DATE')['OPT_WEIGHT'].sum()
        if weight_daily.max() > 2:
            self.portfolio['OPT_WEIGHT'] = self.portfolio['OPT_WEIGHT'] / 100

        # 3. 调整日期格式
        date_type = self.portfolio['DATE'].iloc[0]
        try:
            # 先调成dt.date()格式 再调整非交易日的日期
            if type(date_type) == str or type(date_type) == np.int64:
                self.portfolio['DATE'] = pd.to_datetime(self.portfolio['DATE'].astype('str'))
            elif type(date_type) == dt.datetime or type(date_type) == pd._libs.tslibs.timestamps.Timestamp:
                self.portfolio['DATE'] = self.portfolio['DATE']
        except:
            raise Exception('日期格式不对，请检查')

        # 4. 计算交易日并筛选对应日期的持仓信息
        if not select_date:
            trade_date = self.portfolio[['DATE']].drop_duplicates()
            if trade_type == 'next_day':
                trade_date['trade_date'] = trade_date['DATE'].apply(
                    lambda x: self.trade_cal[self.trade_cal['DATE'] > x.strftime('%Y%m%d')]['DATE'].iloc[0])
            elif trade_type == 'in_day':
                trade_date['trade_date'] = trade_date['DATE'].apply(
                    lambda x: self.trade_cal[self.trade_cal['DATE'] >= x.strftime('%Y%m%d')]['DATE'].iloc[0])
            else:
                raise TypeError('输入的trade_type有误')

            self.portfolio = pd.merge(self.portfolio, trade_date, on='DATE', how='left').drop('DATE', axis=1).rename(
                columns={'trade_date': 'DATE'})

        else:
            portfolio_date = self.portfolio['DATE'].drop_duplicates().sort_values()

            cal_date = pd.date_range(self.portfolio['DATE'].min(), self.portfolio['DATE'].max(), freq=freq)
            cal_date = [x.strftime('%Y%m%d') for x in cal_date]
            tradedate_df = pd.DataFrame({'nominal_trade_date': cal_date})
            tradedate_df['trade_date'] = tradedate_df['nominal_trade_date'].apply(
                lambda x: self.trade_cal[self.trade_cal['DATE'] > x]['DATE'].iloc[0])
            tradedate_df.drop_duplicates(subset=['trade_date'], inplace=True)

            while True:
                n = tradedate_df.shape[0]
                tradedate_df['position'] = tradedate_df['trade_date'].apply(
                    lambda x: self.trade_cal[self.trade_cal['DATE'] == x].index[0])
                tradedate_df['interval'] = tradedate_df['position'].diff().fillna(5.0)
                tradedate_df = tradedate_df[tradedate_df['interval'] > 2]
                if tradedate_df.shape[0] == n:
                    break
                else:
                    pass

            if trade_type == 'in_day':
                tradedate_df['DATE'] = tradedate_df['trade_date']
            else:
                tradedate_df['DATE'] = tradedate_df['trade_date'].apply(
                    lambda x: portfolio_date[portfolio_date < x].iloc[-1])
                tradedate_df = tradedate_df.drop_duplicates(subset=['DATE'], keep='first')

            self.portfolio = pd.merge(self.portfolio, tradedate_df, on=['DATE'], how='right')
            self.portfolio = self.portfolio[['TICKER', 'trade_date', 'OPT_WEIGHT']].rename(
                columns={'trade_date': 'DATE'})

        if len(black_tickers) > 0:
            black_date = black_tickers['black_date'].drop_duplicates()
            black_tickers['sig'] = 1
            self.portfolio['black_date'] = self.portfolio['DATE'].apply(lambda x: black_date[black_date <= x].max())
            self.portfolio = pd.merge(self.portfolio, black_tickers, on=['black_date', 'TICKER'], how='left')
            self.portfolio = self.portfolio[self.portfolio['sig'].isna()].drop('sig', axis=1)
            self.portfolio['OPT_WEIGHT'] = self.portfolio.groupby('DATE').apply(lambda x: x['OPT_WEIGHT'] /
                                                                                          x[
                                                                                              'OPT_WEIGHT'].sum()).reset_index(
                level='DATE', drop=True)

        self.portfolio = self.portfolio[(self.portfolio['DATE'] >= self.start) & (self.portfolio['DATE'] <= self.end)]
        # print(self.portfolio['DATE'].unique())
        if del_warning:
            ori_weight = self.portfolio.groupby('DATE')['OPT_WEIGHT'].sum()
            self.portfolio = pd.merge(self.portfolio, self.st_status, on=['TICKER', 'DATE'], how='left')
            self.portfolio = self.portfolio[self.portfolio['ST_status'] != 1]
            self.portfolio = self.portfolio.drop('ST_status', axis=1)
            self.portfolio['OPT_WEIGHT'] = self.portfolio.groupby('DATE').apply(
                lambda x: x['OPT_WEIGHT'] / x['OPT_WEIGHT'].sum() * ori_weight.loc[x['DATE'].iloc[0]]).reset_index(
                level='DATE', drop=True)
        self.portfolio['OPT_WEIGHT'] = self.portfolio['OPT_WEIGHT'].fillna(0)
        print("", "Portfolio Loaded".center(48, '-'), "")
        self.freq = freq

    def load_price_data(self):

        if hasattr(self, 'price_whole'):
            return

        # df_ticker = pd.DataFrame()
        # if '股票' in self.test_goods:
        print('---Start Loading Ticker Data for the Portfolio---')
        # df_ticker = feather.read_dataframe(self.data_path + 'daily.feather',
        #                                   columns=['TICKER', 'DATE', 'adj_open', 'adj_close', 'adj_pre_close',
        #                                            'trade_status', 'adj_factor', 'avg_price', 'ST_status'])

        df_ticker = feather.read_dataframe(os.path.join(self.data_path, 'daily.feather'),
                                           columns=['TICKER', 'DATE', 'open', 'close', 'amount', 'volume'])
        df_adj = feather.read_dataframe(os.path.join(self.data_path, 'adj_factors.feather'))
        st = feather.read_dataframe(os.path.join(self.data_path, "st_all.feather"))

        df_ticker = pd.merge(df_ticker,df_adj,how='left',on=['TICKER','DATE'])
        df_ticker = pd.merge(df_ticker,st,how='left',on=['TICKER','DATE'])
        df_ticker['ST_status'] = df_ticker['ST_status'].fillna(0)
        df_ticker['open'] = df_ticker['open']*df_ticker['adj_factors']
        df_ticker['close'] = df_ticker['close']*df_ticker['adj_factors']
        df_ticker['pre_close'] = df_ticker.groupby('TICKER')['close'].shift(1)
        df_ticker['avg_price'] = df_ticker['amount']/df_ticker['volume']
        df_ticker['st'] = df_ticker['ST_status']

        df_ticker = df_ticker[['TICKER','DATE','open', 'close', 'pre_close','st','adj_factors','avg_price','ST_status']]
        df_ticker.columns = ['TICKER', 'DATE', 'open', 'close', 'pre_close', 'st', 'adj_factor', 'avg_price',
                             'ST_status']
        vwap_type = [x for x in self.price_type if 'vwap' in x]
        if len(vwap_type) > 0:
            vwap_type = vwap_type[0]
            self.price_type.remove(vwap_type)
            self.price_type.append('vwap')
            vwap_time = vwap_type[5:]
            if vwap_time == '0':
                df_ticker['vwap'] = df_ticker['open']
                pass
            else:
                vwap_name = f'vwap_{vwap_time}.feather'
                vwap_df = feather.read_dataframe(os.path.join(self.data_path,f'{vwap_name}'))
                # vwap_df = vwap_df.rename(columns = {'vwap':vwap_type})
                df_ticker = pd.merge(df_ticker, vwap_df, on=['TICKER', 'DATE'], how='left')
                df_ticker['vwap'] = df_ticker[f'vwap_{vwap_time}'] * df_ticker['adj_factor']

        df_ticker = pd.merge(self.trade_date, df_ticker, on='DATE', how='left')
        df_ticker = df_ticker.sort_values(by=['TICKER', 'DATE'], ascending=True).reset_index(drop=True)
        # df_ticker['st'] = df_ticker['st'].map(lambda x: 0 if x == '停牌' else 1)
        df_ticker['avg_price'] = df_ticker['avg_price'] * df_ticker['adj_factor']

        price_df = df_ticker
        # print(price_df)
        self.st_status = price_df[['TICKER', 'DATE', 'ST_status']]
        price_df = price_df.drop('ST_status', axis=1)

        if self.hedge_index.upper() == 'NO':
            index_df = pd.read_csv(os.path.join(self.data_path,'index_data','index_000300.SH.csv'), dtype={'DATE': str})
            index_df['close'] = 1
            index_df['open'] = 1
        else:
            index_df = pd.read_csv(os.path.join(self.data_path,'index_data',f'index_{self.hedge_index}.csv'), dtype={'DATE': str})

        index_df = index_df.set_index('DATE')
        index_df['st'] = 1
        index_df['avg_price'] = index_df['close']
        self.price_all = {}

        for each in self.price_type:
            df = price_df[['TICKER', 'DATE'] + [each]]
            df = df.pivot_table(columns='TICKER', values=each, index='DATE')

            if each == 'vwap':
                df = df.apply(lambda x: x.fillna(method='ffill'))
                df[self.hedge_index] = index_df['open']
                df = df[(df.index <= self.end)]
                self.price_all[each] = df
            else:
                df = pd.merge(df, index_df[[each]], left_index=True, right_index=True, how='left')
                df[self.hedge_index] = df[each]
                df = df[(df.index <= self.end)]
                self.price_all[each] = df.drop(columns=each)
        print("", "Loading Price Data Finished".center(48, '-'), "")
        return

    def adjust_portfolio(self, st=False, del_warning=False):
        """
        调整组合权重，处理 ST/停牌 股票的情况
        st=True 表示启用 ST 调整
        """
        # 生成初始权重矩阵
        self.portfolio_cal = self.portfolio.pivot_table(values='OPT_WEIGHT', columns='TICKER', index='DATE')
        self.portfolio_cal[self.hedge_index] = self.portfolio_cal.sum(axis=1)

        if st:
            # df_st: 0=正常可交易, 1=停牌
            df_st = self.price_all['st'][self.ticker_pool]

            start_weight = None
            for day in self.portfolio_cal.index:
                daily_weight = self.portfolio_cal.loc[day].copy().drop(self.hedge_index).fillna(0)
                daily_st = df_st.loc[day].copy().drop(self.hedge_index)

                if start_weight is None:
                    # 第一天：把停牌股票剔掉
                    tradable_weight = daily_weight * (daily_st == 0)
                    if tradable_weight.sum() > 0:
                        tradable_weight = tradable_weight / tradable_weight.sum()
                    else:
                        tradable_weight = daily_weight  # 如果全停牌，回退到原始权重
                    tradable_weight[self.hedge_index] = tradable_weight.sum()
                    self.portfolio_cal.loc[day] = tradable_weight
                    start_weight = tradable_weight.drop(self.hedge_index)
                else:
                    # 保留前一天停牌股票的权重
                    frozen_weight = start_weight * (daily_st == 1)  # 停牌股票沿用昨天权重
                    tradable_weight = daily_weight * (daily_st == 0)  # 可交易股票用今天权重

                    # 保证权重总和一致
                    total_weight = tradable_weight.sum() + frozen_weight.sum()
                    if tradable_weight.sum() > 0:
                        tradable_weight = tradable_weight / tradable_weight.sum() * (total_weight - frozen_weight.sum())

                    # 合并两部分
                    daily_weight_adj = tradable_weight.add(frozen_weight, fill_value=0)
                    daily_weight_adj[self.hedge_index] = daily_weight_adj.sum()
                    self.portfolio_cal.loc[day] = daily_weight_adj
                    start_weight = daily_weight_adj.drop(self.hedge_index)

        # 对齐交易日并前向填充
        self.portfolio_cal = pd.merge(
            self.trade_date.set_index('DATE'),
            self.portfolio_cal.fillna(0),
            left_index=True, right_index=True, how='left'
        )
        self.portfolio_cal = self.portfolio_cal.fillna(method='ffill')[
            self.portfolio_cal.index >= self.portfolio['DATE'].min()
            ]

    # def adjust_portfolio(self, st=False, del_warning=False):
    #     # df_st = np.abs(self.price_all['st'].drop(columns = self.hedge_index) - 1)
    #     # 在df_st中，1永远代表不停牌 0代表停牌
    #
    #     # self.portfolio = pd.merge(self.portfolio, self.trade_cal, on = 'DATE', how = 'left').drop(
    #     #     columns = ['Exchange', 'is_open'])
    #     self.portfolio_cal = self.portfolio.pivot_table(values='OPT_WEIGHT', columns='TICKER', index='DATE')
    #     self.portfolio_cal[self.hedge_index] = self.portfolio_cal.sum(axis=1)
    #
    #     if st:
    #         df_st = self.price_all['st'][self.ticker_pool]
    #         for day in self.portfolio_cal.index:
    #             # if day == '20190311':
    #             #     break
    #             daily_weight = self.portfolio_cal.loc[day].copy('deep').drop(self.hedge_index).fillna(0)
    #             daily_st = df_st.loc[day].copy('deep').drop(self.hedge_index)
    #             if day == self.portfolio_cal.index[0]:
    #                 daily_weight_adj = daily_weight * (1-daily_st)  # 停牌股票的权重变为0
    #                 # total_weight = daily_weight_adj.sum()
    #                 daily_weight_adj = daily_weight_adj / daily_weight_adj.sum() * 1
    #                 # daily_weight_adj[self.hedge_index] = total_weight  # 设置当日指数的权重为可交易股票的总权重
    #                 daily_weight_adj[self.hedge_index] = daily_weight_adj.sum()  # 设置当日指数的权重为可交易股票的总权重
    #                 self.portfolio_cal.loc[day] = daily_weight_adj
    #                 start_weight = daily_weight_adj.drop(self.hedge_index)  # 重新分配权重 总权重仍为1
    #             else:
    #                 start_weight = start_weight * np.abs(1-daily_st)  # 停牌的股票，保留权重，可以正常交易的股票，去除权重
    #                 start_weight = start_weight.dropna()[start_weight.dropna() != 0]
    #
    #                 ori_total_weight = daily_weight.sum()  # 原本这一天的总权重，如果空仓的话，就是0
    #                 total_weight = start_weight.sum() * ori_total_weight
    #                 # daily_weight = daily_weight.drop(list(start_weight.index)).append(start_weight)
    #                 # 把已持仓 但因停牌无法交易的股票的权重改为上一期权重
    #
    #                 daily_weight_adj = daily_weight.drop(list(start_weight.index))
    #                 daily_weight_adj = daily_weight_adj / daily_weight_adj.sum() * total_weight  # 计算剩余股票应对应的权重 结果加上已持仓的停牌股票权重仍未100
    #
    #                 daily_weight_adj = daily_weight_adj * daily_st.drop(list(start_weight.index))
    #                 daily_weight_adj = daily_weight_adj / daily_weight_adj.sum() * total_weight
    #
    #                 daily_weight_adj = daily_weight_adj._append(start_weight)
    #                 daily_weight_adj[self.hedge_index] = daily_weight_adj.sum()
    #                 self.portfolio_cal.loc[day] = daily_weight_adj.fillna(0)
    #
    #                 start_weight = daily_weight_adj.fillna(0)
    #
    #     # create a pivot table for portfolio to be used in further calculation
    #     self.portfolio_cal = pd.merge(self.trade_date.set_index('DATE'), self.portfolio_cal.fillna(0),
    #                                   left_index=True, right_index=True, how='left')
    #     self.portfolio_cal = self.portfolio_cal.fillna(method='ffill')[
    #         self.portfolio_cal.index >= self.portfolio['DATE'].min()]

    def run(self, fee=0.0015):
        warnings.filterwarnings('ignore')
        print("", "Strategy Start".center(48, '-'), "")
        print("", "Start-Up Fund is {0}".format(self.start_cash).center(48, '-'), "")
        print("", "Portfolio is {0}".format(self.file_name).center(48, '-'), "")

        price_use = [x for x in self.price_type if x in ['close', 'open']]
        assert len(price_use) == 1
        price_use = price_use[0]
        price_df = self.price_all[price_use][self.ticker_pool]
        price_df = price_df[price_df.index >= self.portfolio['DATE'].min()]

        return_df = price_df.pct_change()
        return_ticker = return_df.drop(columns=self.hedge_index)
        return_index = return_df[[self.hedge_index]]

        ticker_return_daily = return_ticker * self.portfolio_cal.drop(columns=self.hedge_index)
        # ticker_return_daily.dropna(how = 'all', inplace = True)
        ticker_return_daily = ticker_return_daily.sum(axis=1, min_count=1) + 1

        index_return_daily = return_index * self.portfolio_cal[[self.hedge_index]]
        index_return_daily = index_return_daily[self.hedge_index] + 1

        # 在每次调仓日，我们要做出修正
        if price_use == 'open':
            turnover_rate = []
            trade_return = pd.DataFrame(columns=['overnight', 'inday'])
            for day in self.portfolio['DATE'].unique():
                date = (pd.to_datetime(day) - dt.timedelta(days=1)).strftime('%Y%m%d')
                return_today = return_ticker.loc[day]

                if date <= self.portfolio_cal.index.min():
                    ticker_return_daily.loc[day] = index_return_daily.loc[day]
                    continue
                while True:
                    try:
                        df = self.portfolio_cal.loc[date]
                        break
                    except:
                        date = (pd.to_datetime(date) - dt.timedelta(days=1)).strftime('%Y%m%d')
                        # yesterday is not a trade-date
                port_today = self.portfolio_cal.loc[day]
                turnover_today = (port_today - df).abs().sum()
                turnover_rate.append(turnover_today)
                fee = turnover_today / 2 * 0.0015 if self.freq == 'D' else 0.0015
                ticker_return_daily[day] = np.sum(return_today * df.drop(self.hedge_index)) - fee + 1
                index_return_daily[day] = return_index.loc[day] * df[self.hedge_index] + 1
                # trade_return.loc[day] = [ticker_overnight_sum,ticker_inday_sum]

        elif price_use == 'close' and 'avg_price' in self.price_type:
            avg_price = self.price_all['avg_price'][self.ticker_pool]

            inday_return = price_df / avg_price - 1  # 日内收益
            overnight_return = avg_price / price_df.shift(1) - 1
            turnover_rate = []
            for day in self.portfolio['DATE'].unique():
                today_port = self.portfolio_cal.loc[day]

                ticker_inday_today = inday_return.loc[day][return_ticker.columns]
                index_inday_today = inday_return.loc[day][self.hedge_index]

                ticker_inday_sum = np.sum(ticker_inday_today * today_port.drop(self.hedge_index)) + 1
                index_inday_sum = index_inday_today * today_port[self.hedge_index] + 1

                date = (pd.to_datetime(day) - dt.timedelta(days=1)).strftime('%Y%m%d')
                if date <= self.portfolio_cal.index.min():
                    ticker_return_daily.loc[day] = ticker_inday_sum
                    index_return_daily.loc[day] = index_inday_sum
                    continue

                while True:
                    try:
                        df = self.portfolio_cal.loc[date]
                        break
                    except:
                        date = (pd.to_datetime(date) - dt.timedelta(days=1)).strftime('%Y%m%d')

                ticker_overnight_today = overnight_return.loc[day][return_ticker.columns]
                ticker_overnight_sum = np.sum(ticker_overnight_today * df.drop(self.hedge_index)) + 1
                ticker_return_today = ticker_overnight_sum * ticker_inday_sum

                turnover_today = (today_port - df).abs().sum()
                turnover_rate.append(turnover_today)
                fee = turnover_today / 2 * 0.0015 if self.freq == 'D' else 0.0015

                ticker_return_daily[day] = ticker_return_today - fee
                index_return_daily[day] = return_index.loc[day] * df[self.hedge_index] + 1

        elif self.price_type == ['close', 'st']:
            turnover_rate = []
            for day in self.portfolio['DATE'].sort_values().unique():
                today_port = self.portfolio_cal.loc[day]

                date = (pd.to_datetime(day) - dt.timedelta(days=1)).strftime('%Y%m%d')
                if date <= self.portfolio_cal.index.min():
                    ticker_return_daily.loc[day] = 1
                    index_return_daily.loc[day] = 1
                    continue

                while True:
                    try:
                        df = self.portfolio_cal.loc[date]
                        break
                    except:
                        date = (pd.to_datetime(date) - dt.timedelta(days=1)).strftime('%Y%m%d')

                turnover_today = (today_port - df).abs().sum()
                turnover_rate.append(turnover_today)
                fee = turnover_today / 2 * 0.0015 if self.freq == 'D' else 0.0015

                # 上一期持仓的close return
                return_today = return_ticker.loc[day]
                last_port_return = return_today * df.drop(self.hedge_index)
                ticker_return_daily[day] = last_port_return.sum() - fee + 1
                index_return_daily[day] = return_index.loc[day] * df[self.hedge_index] + 1

        elif 'vwap' in self.price_type:
            avg_price = self.price_all['vwap'][self.ticker_pool]

            inday_return = price_df / avg_price - 1  # 日内收益
            overnight_return = avg_price / price_df.shift(1) - 1
            turnover_rate = pd.Series()
            trade_return = pd.DataFrame(columns=['overnight', 'inday'])
            for day in self.portfolio['DATE'].unique():
                today_port = self.portfolio_cal.loc[day]

                ticker_inday_today = inday_return.loc[day][return_ticker.columns]
                index_inday_today = inday_return.loc[day][self.hedge_index]

                ticker_inday_sum = np.sum(ticker_inday_today * today_port.drop(self.hedge_index)) + 1
                index_inday_sum = index_inday_today * today_port[self.hedge_index] + 1

                date = (pd.to_datetime(day) - dt.timedelta(days=1)).strftime('%Y%m%d')
                if date <= self.portfolio_cal.index.min():
                    ticker_return_daily.loc[day] = ticker_inday_sum
                    index_return_daily.loc[day] = index_inday_sum
                    continue

                while True:
                    try:
                        df = self.portfolio_cal.loc[date]
                        break
                    except:
                        date = (pd.to_datetime(date) - dt.timedelta(days=1)).strftime('%Y%m%d')

                ticker_overnight_today = overnight_return.loc[day][return_ticker.columns]
                ticker_overnight_sum = np.sum(ticker_overnight_today * df.drop(self.hedge_index)) + 1
                ticker_return_today = ticker_overnight_sum * ticker_inday_sum

                turnover_today = (today_port - df).abs().sum()
                # turnover_rate.append(turnover_today)
                turnover_rate.loc[day] = turnover_today
                trade_return.loc[day] = [ticker_overnight_sum, ticker_inday_sum]
                fee = turnover_today / 2 * 0.0015 if self.freq == 'D' else 0.0015

                ticker_return_daily[day] = ticker_return_today - fee
                index_return_daily[day] = return_index.loc[day] * df[self.hedge_index] + 1

        else:
            raise TypeError('输入的price_type有错')
        hedge_return = ticker_return_daily - index_return_daily + 1
        hedge_return.iloc[0] = 1
        ticker_return_daily.iloc[0] = 1
        index_return_daily.iloc[0] = 1

        self.hedge_return = hedge_return.sort_index().expanding(1).apply(np.prod)
        self.port_return = ticker_return_daily.sort_index().expanding(1).apply(np.prod)
        self.index_return = index_return_daily.sort_index().expanding(1).apply(np.prod)
        self.turnover = np.nanmean(turnover_rate)

        # if trade_return.empty:
        #     pass
        # self.trade_day_return = trade_return.reset_index().rename(columns = {'index':'DATE'})

        df_ret = pd.DataFrame(hedge_return.rename('Hedge_Daily'))
        df_ret['Port_Daily'] = ticker_return_daily
        df_ret['Index_Daily'] = index_return_daily
        self.df_ret = df_ret

    def draw(self, start, end, pic_folder, pic_name):
        date_drop = self.portfolio_cal[self.portfolio_cal[self.hedge_index] == 0].index
        if len(date_drop) > 0:
            print('存在权重为0日期')

        date_list = self.hedge_return.reset_index()['DATE']
        date_list = date_list[(date_list >= start) & (date_list <= end)]

        index_series = self.index_return[date_list.to_list()]
        port_series = self.port_return[date_list.to_list()]
        hedge_series = self.hedge_return[date_list.to_list()]
        hedge_return_daily = self.df_ret['Hedge_Daily'][date_list.to_list()]

        draw_pic(date_list, hedge_return_daily, index_series, port_series, hedge_series, pic_folder, pic_name,
                 turnover=self.turnover, date_drop=date_drop)

    def num_result(self, data_name, num_folder, start, end, num_name):
        warnings.filterwarnings('ignore')
        date_list = self.hedge_return.reset_index()['DATE']
        date_list = date_list[(date_list >= start) & (date_list <= end)]
        hedge_return_daily = self.df_ret['Hedge_Daily'][date_list.to_list()]
        daily_ret = hedge_return_daily - 1

        sr = sharpe_ratio(daily_ret)
        ar = annual_return(daily_ret)
        md = max_drawdown(daily_ret)
        cm = calmar_ratio(daily_ret)

        num = pd.DataFrame(
            {'portfolio': [self.file_name], 'sharpe_ratio': [sr], 'annual_return': [ar], 'max_drawdown': [md],
             'calmar_ratio': [cm], 'start': [start],
             'end': [end], 'flag': ['all']},
            index=[data_name])
        # 保留每个组合的sharp ratio, annual return 和 最大回撤数据

        daily_return = pd.DataFrame(
            {'Hedged_Return': self.hedge_return, 'Port_Return': self.port_return, 'Index_Return': self.index_return})
        daily_return = pd.merge(daily_return, self.df_ret, left_index=True, right_index=True, how='inner')

        year_list = [x.strftime("%Y%m%d") for x in pd.date_range(start, end, inclusive='left', freq='Y')
                     if (x.strftime("%Y%m%d") > start)]
        year_list = [start] + year_list + [end]
        abnormal_list = [('20230322', '20230430'), ('20240201', '20240207'), ('20240415', '20240416'),
                         ('20240426', '20240429'), ('20221030', '20221231'), ('20200828', '20200916'),
                         ('20240603', '20240606'), ('20240924', '20241009'),('20250406','20250415')]
        for i in range(1, len(year_list)):
            start_temp, end_temp = year_list[i - 1:i + 1]
            date_list_temp = date_list[(date_list >= start_temp) & (date_list <= end_temp)]
            hedge_return_daily_temp = self.df_ret['Hedge_Daily'][date_list_temp.to_list()] - 1

            sr = sharpe_ratio(hedge_return_daily_temp)
            ar = annual_return(hedge_return_daily_temp)
            md = max_drawdown(hedge_return_daily_temp)
            cm = calmar_ratio(hedge_return_daily_temp)
            num.loc[i] = [self.file_name, sr, ar, md, cm, start_temp, end_temp, end_temp]

        for start_temp, end_temp in abnormal_list:
            date_list_temp = date_list[(date_list >= start_temp) & (date_list <= end_temp)]
            hedge_return_daily_temp = self.df_ret['Hedge_Daily'][date_list_temp.to_list()] - 1

            sr = sharpe_ratio(hedge_return_daily_temp)
            ar = annual_return(hedge_return_daily_temp)
            md = max_drawdown(hedge_return_daily_temp)
            cm = calmar_ratio(hedge_return_daily_temp)
            num.loc[start_temp] = [self.file_name, sr, ar, md, cm, start_temp, end_temp, f'{start_temp}_{end_temp}']

        self.ratio_result = pd.concat([self.ratio_result, num])
        num_folder = num_folder or 'E:/QuantWork/XCSX/Result_data/'
        try:
            daily_return.reset_index().to_csv(os.path.join(num_folder, f'Daily_Return_{num_name}.csv'))
        except:
            daily_return.reset_index().to_csv(os.path.join(num_folder, f'{num_name}.csv'))

        # self.trade_day_return.to_csv(os.path.join(num_folder, f'Trade_Date_Return_{num_name}.csv'))
        # 保留每日的return结果

    def save_result(self, result_save_folder, ratio_name=None, split=True):
        if split:
            assert ratio_name is not None, '指标结果输出excel为空'
            ratio_name = 'Ratio_Result_{0}.xlsx'.format(ratio_name)
            self.ratio_result.to_excel(os.path.join(result_save_folder, ratio_name))
        else:
            today = dt.datetime.today().strftime('%Y%m%d')
            suffix = result_save_folder.replace('\\', '_')
            excel_name = f'{suffix}_{today}_回测指标汇总.xlsx'
            if os.path.exists(os.path.join(result_save_folder, excel_name)):
                wb = op.load_workbook(os.path.join(result_save_folder, excel_name))
                for items in self.ratio_result.values:
                    wb['回测结果'].append(list(items))
                wb.save(os.path.join(result_save_folder, excel_name))
                wb.close()
            else:
                self.ratio_result.to_excel(os.path.join(result_save_folder, '指标结果.xlsx'),
                                           index=False, sheet_name='回测结果')

    def __call__(self, folder=None, file_name='test_portfolio.csv', st=False, result_save_folder=None,
                 draw_pic=False, hedge='399905.SZ', params=None, trade_params=None, draw_parms=(None, None),
                 result_name=None, select_date=False, freq='M', trade_type='next_day', fee=0.0015, del_warning=False,
                 black_list=None):

        args = params or {'start_cash': 10000000}
        trade_args = trade_params or {'price_type': 'open', 'proportion': 1}
        draw_parms = draw_parms if all(draw_parms) else (self.start, self.end)

        self.set_params(args)
        self.set_hedge(hedge)
        self.set_trade_params(trade_args)
        self.get_trade_date()

        self.load_price_data()

        self.load_portfolio(folder=folder, file_name=file_name, select_date=select_date, freq=freq,
                            trade_type=trade_type, del_warning=del_warning, black_list=black_list)
        self.set_pool()
        self.adjust_portfolio(st=st)
        self.run(fee=fee)

        result_save_folder = result_save_folder or folder
        if not os.path.exists(result_save_folder):
            os.makedirs(result_save_folder)
        else:
            pass

        if draw_pic:
            self.draw(start=draw_parms[0], end=draw_parms[1], pic_folder=result_save_folder, pic_name=result_name)
        else:
            pass

        result_name = result_name if not hasattr(self, 'replace_name') else self.replace_name
        self.num_result(data_name=file_name.replace('.csv', ''), start=draw_parms[0], end=draw_parms[1],
                        num_folder=result_save_folder, num_name=result_name)


if __name__ == '__main__':
    today = dt.datetime.today().strftime('%Y%m%d')

    #  生成回测实例  这里的start和end就是回测的起止日期

    start_date, end_date = '20170301', '20220930'

    # select_date = False  # 这个参数是指，我们是否要从权重文件中，选择指定的一些日期作为权重计算日。
    # 比如权重文件是日频的，我们按照月频交易，那么我们就需要选出这些天的权重数据，
    # 如果权重文件中全部日期都是要进行权重计算并交易的，就没必要再选指定日期，直接False就行

    # freq = 'M'  # 交易频率 月频 M 周频 W
    trade_type = 'next_day'  # 当天盘中交易是 in_day， 隔天交易是 next_day
    use_info = get_Multi_info('请输入测试参数', [('单选', '请选择对冲指数\n(不对冲选择No)',
                                                  ['399905.SZ', '000852.SH', '8841431.WI', '000300.SH', 'No']),
                                                 ('input', '请输入起始日期(默认20230104)', ''),
                                                 ('input', '请输入截止日期(默认20250822)', ''),
                                                 ('单选', '请选择测试频率', ['D', 'M']),
                                                 ('单选', '请选择测试标的', ['股票']),
                                                 (
                                                 '单选', '请选择收益价格', ['open', 'close-vwap', 'close', 'close-open',
                                                                            'close-vwap_930_1000', 'close-vwap_930_1030',
                                                                            'close-vwap_1000_1030',
                                                                            'close-vwap0', 'close-vwaplast30minutes']),
                                                 ('单选', '请选择是否调整停牌', ['Y', 'N']),
                                                 ('单选', '请选择是剔除ST股票', ['Y', 'N']),
                                                 ('单选', '请选择是否挑选日期', ['Y', 'N']),
                                                 ('input', '请输入手续费(默认0.0015)', '')])
    hedge, start_date, end_date, freq, test_goods, price_type, st, del_warning, select_date, fee = use_info
    start_date = start_date or '20230104'
    end_date =  end_date or '20250822'
    hedge = hedge or '000852.SH'
    fee = fee or '0.0015'
    fee = float(fee)
    # test_goods = ['股票']

    trade_params = {'proportion': 1,
                    'price_type': ['close', 'avg_price'] if price_type == 'close-vwap'
                    else ['open'] if price_type == 'open' else ['close', 'vwap_930_1130'] if price_type == 'close-vwap12'
                    else ['close', 'vwap_930_1000'] if price_type == 'close-vwap_930_1000'
                    else ['close', 'vwap_930_1030'] if price_type == 'close-vwap_930_1030'
                    else ['close', 'vwap_1000_1030'] if price_type == 'close-vwap_1000_1030'
                    else ['close', 'vwap_0'] if price_type == 'close-vwap0'
                    else ['close', 'vwap_1430_1500'] if price_type == 'close-vwaplast30minutes'
                    else ['close', 'open'] if price_type == 'close-open'
                    else ['close']}

    path_info = get_Multi_info('请输入路径信息', [('input', '请输入组合文件路径', ''),
                                                  ('input', '请输入结果输出路径', ''),
                                                  ('input', '请输入黑名单因子个数', ''),
                                                  ('单选', '组合文件类型', ['.csv', '.feather', '.xlsx'])])
    folder, result_folder, black_number, port_file_type = path_info

    assert len(folder) * len(result_folder) > 0, '输入路径为空'

    st = True if st == 'Y' else False
    del_warning = True if del_warning == 'Y' else False
    select_date = True if select_date == 'Y' else False
    black_list = rf'\\10.36.35.85\Data_Storage\Financial_Factors\财务因子\中信负向组合\factors\final_sum_min_counts_{black_number}_非金融.feather' if len(
        black_number) > 0 else None

    os.makedirs(result_folder, exist_ok=True)
    file_name_list = getSelection('请选择测试组和', [x for x in os.listdir(folder) if port_file_type in x], row_count=2)
    result_name_list = [x.replace('.feather', '_') +
                        f'{freq}_{hedge}_st.{int(st)}_选天.{int(select_date)}_价格_{price_type}_{dt.datetime.today().strftime("%H%M%S")}'
                        for x in file_name_list]
    back_test_terminal_1 = back_test(start=start_date, end=end_date, test_goods=test_goods)
    self = back_test_terminal_1
    for i in range(len(file_name_list)):
        ###################注意 DATE 列 要改成'20100101'这种8位字符串格式
        file_name = file_name_list[i]
        result_name = result_name_list[i]
        # 主函数
        os.makedirs(os.path.join(result_folder,file_name),exist_ok=True)
        back_test_terminal_1(folder=folder, file_name=file_name, draw_pic=True, st=st,
                             result_save_folder=os.path.join(result_folder,file_name), trade_params=trade_params,
                             result_name=result_name, select_date=select_date, freq=freq, hedge=hedge, fee=fee,
                             del_warning=del_warning, black_list=black_list)

        #  输出回测的数据结果
        #  之所以单独加这一步而不是再回测结束后自动输出，是因为我们可以在前面多做几个回测，在这里统一输出数据结果
    # self.save_result(result_save_folder=result_folder, split=False)
        self.save_result(result_save_folder=os.path.join(result_folder,file_name), split=False)
