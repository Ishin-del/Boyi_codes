from __future__ import division
from __future__ import print_function
import numpy as np
import pandas as pd
import feather
import datetime as dt
import statsmodels.api as sm
from tqdm import tqdm
import os
from statsmodels.stats.weightstats import DescrStatsW
from scipy.special import erfinv as erf
import warnings
# from xzl_func import my_func
# import Tool_box.cyy_funcs as my_func
from joblib import Parallel, delayed
import psutil
from test_tools.SignalProcessor import ProcessFactor
import datetime
import tkinter as tk
from tkinter import ttk, messagebox

warnings.simplefilter('ignore', FutureWarning)


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


def kill_mutil_process(pid_list=[]):
    if len(pid_list) == 0:
        current_pids = psutil.pids()
        # 找出使用Python的进程
        python_pids = [pid for pid in current_pids if 'python.exe' in psutil.Process(pid=pid).name()]
        # python_cmd = [psutil.Process(pid=pid).cmdline() for pid in python_pids]  # 查看所有python进程
        python_multil_process = [pid for pid in python_pids if
                                 '--multiprocessing-fork' in str(psutil.Process(pid=pid).cmdline())]
    else:
        python_multil_process = pid_list
    for pid in python_multil_process:
        try:
            # print('Killing process with pid {}'.format(pid))
            psutil.Process(pid=pid).terminate()  # 关闭进程
        except:
            continue


def signal_process_func(factor_folder_name, fill_whole_path, n_jobs=3, **kwargs):
    '''
    split_output: 是否分开输出文件内的多个因子
    factor_name_dict: 格式 file:[factor_list] 表示每个文件要process哪几个因子
                      可以为空，则默认process全部因子
    file_name_list: 文件列表

    【out_put_name_dict】 格式 file_name: output_file_name / 当split_out为True时，该dict不能为空
    【out_put_folder_dict】 格式 factor_name: output_file_name / 当split_out为False时，该dict不能为空
    以上两个只能写一个
    数据文件路径
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

    factor_path = fill_whole_path
    if 'out_put_path' in kwargs.keys():
        out_put_path = kwargs['out_put_path']
    else:
        out_put_path = factor_path.replace('补全版本', '中性化数据')

    if 'is_polynomial' in kwargs.keys():
        is_polynomial = kwargs['is_polynomial']
        is_polynomial = 'polynomial' if is_polynomial == '多项式拟合' else 'segment_polynomial' if is_polynomial == '分段多项式拟合' else '无处理'
    else:
        is_polynomial = '无处理'

    if 'signal_start' in kwargs.keys() or 'signal_end' in kwargs.keys():
        signal_start = kwargs['signal_start']
        signal_end = kwargs['signal_end']
    else:
        signal_start = '19000101'
        signal_end = '22001231'

    # 确定一些统一的中性化参数
    stand_type = 'ZSCORE' if 'stand_type' not in kwargs.keys() else kwargs['stand_type']
    linear_param = True if is_polynomial != '无处理' else False
    linear_mode = is_polynomial if linear_param else None

    if linear_param or stand_type != 'ZSCORE':
        suffixes = ''
        suffixes = suffixes if stand_type == 'ZSCORE' else suffixes + '_' + stand_type
        suffixes = suffixes if not linear_param else suffixes + '_' + linear_mode
        head, tail = os.path.split(out_put_path)
        tail = tail + suffixes
        out_put_path = os.path.join(head, tail)
        print(out_put_path)
    else:
        pass

    os.makedirs(out_put_path, exist_ok=True)
    split_output = False

    if 'file_name_list' in kwargs.keys():
        file_name_list = kwargs['file_name_list']
    else:
        file_name_list = os.listdir(factor_path)
    file_name_list = [i for i in file_name_list if '.feather' in i]
    if 'factor_name_dict' in kwargs.keys():
        factor_name_dict = kwargs['factor_name_dict']
    else:
        factor_name_dict = {}  # 格式 file:[factor_list] 表示每个文件要process哪几个因子；可以为空，则默认process全部因子

    if 'base_savepath' in kwargs.keys():
        base_savepath = kwargs['base_savepath']
    else:
        base_savepath = None

    out_put_name_dict = {}  # 这个写了下一条语句就不会运行，如果是空的，则默认在原始文件后面价格processed输出
    out_put_name_dict = {x: x.replace('.feather', '_processed.feather') for x in file_name_list} \
        if len(out_put_name_dict) == 0 else out_put_name_dict
    # out_put_folder_dict = {'vibration_information\\vibration_information_factor.feather': 'vibration_information'}

    data_path = r'Z:\local_data\Data_Storage'  # 这里改成Data_Storage的位置
    processor = ProcessFactor(data_path=data_path, factor_list=[], list_date_limit=30, end=signal_end)
    df_basic_data = processor.load_data(ST=True)  # 1分钟
    print('中性化基础数据加载完毕')
    error_list = {}

    def base_sig_func(file_name):
        pid = os.getpid()
        error_list_1 = {}
        if '.xlsx' in file_name:
            df_orignal_data = pd.read_excel(factor_path + file_name)
            try:
                df_orignal_data['DATE'] = df_orignal_data['DATE'].map(lambda x: x.strftime('%Y%m%d'))
            except:
                try:
                    df_orignal_data['DATE'] = df_orignal_data['DATE'].astype(str)
                except:
                    pass

        elif '.csv' in file_name:
            df_orignal_data = pd.read_csv(factor_path + file_name, dtype={'DATE': str})
        elif '.feather' in file_name:
            '''读取文件在这里操作'''
            df_orignal_data = feather.read_dataframe(os.path.join(factor_path ,file_name))
            # TODO:
            df_orignal_data.replace([np.inf,-np.inf],np.nan,inplace=True)
            df_orignal_data['DATE'] = df_orignal_data['DATE'].astype(str).apply(lambda x: x.replace('-', ''))
            if 'type' in df_orignal_data.columns:
                df_orignal_data = df_orignal_data.drop(columns='type')
            if 'mon50_return' in df_orignal_data.columns:
                df_orignal_data = df_orignal_data.drop(columns='mon50_return')
            if 'ST_status' in df_orignal_data.columns:
                df_orignal_data = df_orignal_data.drop(columns='ST_status')
            if 'trade_status' in df_orignal_data.columns:
                df_orignal_data = df_orignal_data.drop(columns='trade_status')

        else:
            raise TypeError('The type of file is incorrect.')

        df_orignal_data = df_orignal_data[(df_orignal_data['DATE'] >= signal_start) &
                                          (df_orignal_data['DATE'] <= signal_end)]
        try:
            factor_name_list = factor_name_dict[file_name]
        except:
            factor_name_list = df_orignal_data.columns.drop(['TICKER', 'DATE']).to_list()

        '''剔除ST和新股'''
        '''不剔除新股，list_date_limi=0'''''
        processor.factor_name = factor_name_list
        df_whole = processor.load_factor(df_basic_data, df_orignal_data, ST=True)  # 这里需要3-4分钟
        del df_orignal_data

        # df_result = pd.DataFrame(columns=['TICKER', 'DATE'])
        for i in range(len(factor_name_list)):
            factor_name = factor_name_list[i]
            try:
                already = pd.DataFrame()
                if os.path.exists(os.path.join(base_savepath, file_name.split('.feather')[0] + '_processed.feather')):
                    already = feather.read_dataframe(os.path.join(base_savepath, file_name.split('.feather')[0] + '_processed.feather'))
                    already = already.dropna().reset_index(drop=True)
                    already_max = already['DATE'].max()
                    df_whole = df_whole[df_whole['DATE'] > already_max]
                    if df_whole.empty:
                        feather.write_dataframe(already, os.path.join(out_put_path, out_put_name_dict[file_name]))
                        continue

                df_processed = processor(df_raw_data=df_whole, factor=factor_name, cap_mode='normal',
                                         regress_mode='OLS',
                                         n_draw=1, stand_type=stand_type,
                                         win_type='MAD', fillna_type='median', merge_cap=False, fillna=True)
            except Exception as e:
                print(str(e))
                error_list_1.update({file_name + '/' + factor_name: str(e)})
                continue

            df_processed = df_processed.dropna()
            df_result = pd.concat([already,df_processed])
            df_result = df_result.sort_values(by=['TICKER','DATE']).reset_index(drop=True)
            # df_result = pd.merge(df_result, df_processed, on=['TICKER', 'DATE'], how='outer')

            if not split_output:
                out_put_name = out_put_name_dict[file_name]
                feather.write_dataframe(df_result, os.path.join(out_put_path, out_put_name))
            else:
                pass

        return error_list_1, pid

    '''processing 这里慢 4分钟左右'''
    if len(file_name_list) < 10:
        for j in range(len(file_name_list)):
            error1, _ = base_sig_func(file_name_list[j])
            error_list.update(error1)
    else:
        if n_jobs > 1:
            print('多进程中性化')
            new_data = Parallel(n_jobs=n_jobs)(
                delayed(base_sig_func)(file_name) for file_name in file_name_list)
            pid_list = [i[1] for i in new_data]
            error_ls = [i[0] for i in new_data]
            for error1 in error_ls:
                error_list.update(error1)
            kill_mutil_process(pid_list=list(set(pid_list)))
        else:
            pid_list = []
            error_ls = []
            for file_name in file_name_list:
                res = base_sig_func(file_name)
                error_ls.append(res[1])
                pid_list.append(res[0])
            for error1 in error_ls:
                # error_list.update(error1)
                # print(error1)
                pass
            # print(pid_list)
            # kill_mutil_process(pid_list=list(set(pid_list)))

    # print(error_list)
    return out_put_path, error_list


def signal_processor_terminal(end_date):
    print('开始中性化：')
    is_polynomial = '无处理'
    # info_list = get_Multi_info('请输入中性化参数',
    #                            [
    #                                ('单选', '因子开发人', ['周楷贺', '吴文博', '周子逸', '田逸心', '高嘉伟']),
    #                                ('单选', '进行中性化文件夹',
    #                                 ['拟使用量价因子', '财务日频', '财务', '另类', '另类日频']),
    #                                ('单选', 'loky进程数', ['1', '2', '4', '8', '12'])])
    #
    # researcher, factor_file, jobuse = info_list
    researcher, factor_file, jobuse = '田逸心','拟使用量价因子',1
    # end_Date = input('请输入要中性化到的日期（默认今天）:')
    end_Date = end_date
    end_Date = end_Date if len(end_Date) > 0 else datetime.datetime.today().strftime('%Y%m%d')

    signal_input_path = rf'\\DESKTOP-79NUE61\Factor_Storage\{researcher}\原始数据\{factor_file}'
    # signal_input_path=r'C:\Users\admin\Desktop\test'
    signal_output_path = rf'\\DESKTOP-79NUE61\Factor_Storage\{researcher}\中性化数据\{factor_file}'
    base_savepath = rf'\\DESKTOP-79NUE61\Factor_Storage\{researcher}\中性化数据备份\{factor_file}'
    os.makedirs(base_savepath, exist_ok=True)

    # create_path(signal_output_path)
    # if not re_sig:
    #     exist_files = os.listdir(signal_output_path)
    #     signal_file_ls = [x for x in signal_file_ls if x.replace('.feather', '_processed.feather') not in exist_files]
    out_put_path, error_list = signal_process_func(factor_folder_name=None, fill_whole_path=signal_input_path,
                                                   out_put_path=signal_output_path, signal_start='20200102',
                                                   signal_end=end_Date,base_savepath=base_savepath,
                                                   is_polynomial=is_polynomial, n_jobs=int(jobuse), re_sig=True)
    test_input_path = out_put_path
    # 写入中性化错误文档
    if len(error_list.items()) > 0:
        with open(os.path.join(out_put_path, f'{datetime.datetime.today().strftime("%Y%m%d")}_中性化报错.txt'),
                  'a+') as f:
            f.write('\n')
            f.write(f"【{datetime.datetime.now().strftime('%Y%m%d %H:%M:%M')}】\n")
            for factor, error1 in error_list.items():
                f.write('\t' + factor + ': ' + str(error1) + '\n')
        f.close()
    print('中性化结束')


if __name__ == '__main__':
    signal_processor_terminal('20251103')
