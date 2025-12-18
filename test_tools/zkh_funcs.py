import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
import psutil
import pandas as pd
import datetime as dt
import feather
import calendar
import socket
import glob
import smtplib
from email.mime.text import MIMEText
import sys
from tool_tyx.data_path import FactorPath
sys.path.append(r'../')



def get_current_computer_hostname():
    return socket.gethostname()


def get_month_end(date):
    # date = 20100312
    date = dt.datetime.strptime(str(date), '%Y%m%d')
    a, b = calendar.monthrange(date.year, date.month)
    newdate = dt.datetime(year=date.year, month=date.month, day=b)
    newdate = newdate.strftime('%Y%m%d')
    return newdate

def drop_ST_new(df, new_date=30):
    # new = pd.read_csv(r'\\10.36.35.85\Data_Storage\Basic_Files\ticker_info.csv',
    #                   usecols=['TICKER', 'start_date', 'end_date'],
    #                   dtype={'start_date': str, 'end_date': str})
    new=feather.read_dataframe(r'\\192.168.1.210\Data_Storage2\ticker_info.feather')
    new['start_date'] = new['start_date'].apply(lambda x: (
            pd.to_datetime(x) + dt.timedelta(days=new_date)).strftime('%Y%m%d'))
    df = pd.merge(df, new, on=['TICKER'], how='left').copy()
    df = df.dropna(subset=['start_date'])
    df = df[(df['DATE'] >= df['start_date']) & (df['DATE'] <= df['end_date'])]
    df = df.reset_index(drop=True).drop(columns=['start_date', 'end_date'])
    ST = feather.read_dataframe(#r'\\10.36.35.85\Data_Storage\Factor_Storage\daily.feather',
                                r'\\192.168.1.210\Data_Storage2\ST_info.feather')

    df = pd.merge(df, ST, on=['TICKER', 'DATE'], how='left')

    # 删除当天为ST的股票
    # df = df[df['ST_status'] == 0].sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
    df = df[df['ST_status'] == '否'].sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
    df = df.drop(columns=['ST_status'])
    return df


def get_daily(datatype):
    daily = pd.read_feather(os.path.join(FactorPath.data_storage,r'daily.feather'), columns=['TICKER', 'DATE'])
    daily = drop_ST_new(daily, new_date=30)

    # 财务数据要非金融的
    # if '非财务' in datatype:
    #     pass
    # else:
    #     financial_columns = ['b10l0', 'b10m0', '10000', 'b10u0']
    #     daily = daily[~daily['zx_Citic_Code'].isin(financial_columns)].reset_index(drop=True)

    # 月频数据
    if '月频' in datatype:
        '''1107新增'''
        date_ls = pd.read_csv(r'\\10.36.35.85\Data_Storage\Basic_Files\Calendar.csv', usecols=['DATE'],
                              converters={'DATE': str})
        date_ls['month'] = date_ls['DATE'].str[:6]
        date_ls = date_ls.sort_values(by=['DATE']).reset_index(drop=True)
        date_ls = date_ls.drop_duplicates(subset=['month'], keep='last').reset_index(drop=True)
        '''1107end'''

        daily = daily[daily['DATE'].isin(date_ls['DATE'])]
        daily = daily.sort_values(by=['TICKER', 'DATE']).reset_index(drop=True)
        daily['month'] = daily['DATE'].apply(lambda x: x[:6])
        daily = daily.drop_duplicates(subset=['TICKER', 'month'], keep='last')
        daily = daily.drop(columns='month').reset_index(drop=True)
        daily['DATE'] = daily['DATE'].apply(lambda x: get_month_end(x))
    else:
        pass
    return daily


def read_file(data_path, file_type='feather'):
    if os.path.exists(data_path):
        if file_type == 'feather':
            df = feather.read_dataframe(data_path)
        elif file_type == 'csv':
            try:
                df = pd.read_csv(data_path, dtype={'DATE': str})
            except:
                df = pd.read_csv(data_path, dtype={'DATE': str},encoding='gbk')
        elif file_type == 'xlsx':
            df = pd.read_excel(data_path, dtype={'DATE': str})
        elif file_type == 'h5':
            df = pd.read_hdf(data_path)
            df['DATE'] = df['DATE'].astype(int).astype(str)
        else:
            raise TypeError(f'Wrong file type: {file_type}')
        max_date = df['DATE'].max()
    else:
        print('不存在: ', os.path.split(data_path)[-1])
        df = pd.DataFrame()
        max_date = None

    return df, max_date



def next_date(date, out_type='', date_type='BS'):
    if type(date) == dt.date or type(date) == dt.datetime or type(date) == pd._libs.tslibs.timestamps.Timestamp:
        if date_type == 'BS':
            date_df = pd.read_csv(FactorPath.calender_path, dtype={'trade_date': str})
            date_df = date_df.rename(columns={'trade_date':'DATE'})
            today = date.strftime('%Y%m%d')
            try:
                next_day = pd.to_datetime(date_df['DATE'][date_df['DATE'] > today].iloc[0])
            except:
                print('Need to update the calendar')
                next_day = dt.datetime.today() + dt.timedelta(days=1)
        else:
            next_day = date + dt.timedelta(days=1)

        if out_type == '':
            next_day = next_day.strftime('%Y%m%d')
        elif out_type == '-':
            next_day = next_day.strftime('%Y-%m-%d')
        elif out_type == '/':
            next_day = next_day.strftime('%Y/%m/%d')
        elif out_type == 'int':
            next_day = int(next_day.strftime('%Y%m%d'))
        elif out_type == 'date':
            next_day = next_day.date()
        elif out_type == 'datetiime':
            pass

        return next_day

    elif type(date) == str:
        # 要求格式‘YYYYmmdd'
        if date_type == 'BS':
            date_df = pd.read_csv(FactorPath.calender_path, dtype={'trade_date': str})
            date_df = date_df.rename(columns={'trade_date': 'DATE'})
            try:
                next_day = pd.to_datetime(date_df['DATE'][date_df['DATE'] > date].iloc[0])
            except:
                print('Need to update the calendar')
                next_day = dt.datetime.today() + dt.timedelta(days=1)
        else:
            next_day = pd.to_datetime(date) + dt.timedelta(days=1)

        if out_type == '':
            next_day = next_day.strftime('%Y%m%d')
        elif out_type == '-':
            next_day = next_day.strftime('%Y-%m-%d')
        elif out_type == '/':
            next_day = next_day.strftime('%Y/%m/%d')
        elif out_type == 'int':
            next_day = int(next_day.strftime('%Y%m%d'))
        elif out_type == 'date':
            next_day = next_day.date()
        elif out_type == 'datetiime':
            pass

        return next_day


def zscore(x):
    mean_ = np.nanmean(x)
    std_ = np.nanstd(x, ddof=1)
    x_out = (x - mean_) / std_
    return x_out



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


def checkInfo(content):
    def normal_exit():
        window.destroy()

    window = tk.Tk('请确认信息')
    window.wm_attributes('-topmost', 1)
    width = 500
    height = 500
    screenwidth = window.winfo_screenwidth()
    screenheight = window.winfo_screenheight()
    window.geometry('%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2))

    frame1 = tk.Frame(window)
    frame1.pack(pady=10, padx=10, expand=True)

    if isinstance(content, str):
        lable1 = tk.Label(frame1, height=3)
        lable1['text'] = content
        lable1.grid(ipadx=10, ipady=10)
    elif isinstance(content, list):
        for i in content:
            lable1 = tk.Label(frame1, height=1)
            lable1['text'] = i
            lable1.grid(ipadx=10, ipady=10)

    ttk.Button(frame1, text="确认", command=normal_exit).grid(ipadx=10, ipady=10)

    window.mainloop()


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        pass

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
