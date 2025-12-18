import pandas as pd
import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import feather

# 设置Matplotlib的字体参数（避免中文乱码）
plt.rcParams['font.family'] = 'SimHei'  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 目标收益列（需与输入文件中的列名对应）
# target_columns = [
#     '平均卖出收益_1100_1400',
#     '开盘后10秒vwap收益_1100_1400',
#     '开盘后30秒vwap收益_1100_1400',
#     '开盘后60秒vwap收益_1100_1400',
#     '开盘后180秒vwap收益_1100_1400',
#     '开盘后300秒vwap收益_1100_1400',
#     '开盘后600秒vwap收益_1100_1400',
#     '开盘后1200秒vwap收益_1100_1400',
#     '开盘后1800秒vwap收益_1100_1400'
# ]

target_columns = [
    '平均卖出收益_1100_1400'
]

# 用于分组的排名函数（将0-1的百分比排名分为0-9共10组）
def rank_func(num):
    return np.floor((num - 0.0000001) * 10)


# 设置工作目录为当前文件夹
wkdir = os.getcwd()
# 创建result文件夹（如果不存在）
result_dir = os.path.join(wkdir, 'result')
os.makedirs(result_dir, exist_ok=True)


def pivot_adjusted(df_sz_temp, output_path):
    """生成透视表并保存到指定路径，包含样本数和单日平均收益"""
    # 计算单日平均收益（先按DATE和group求均值，再按group求均值）
    rst_1 = df_sz_temp.groupby(["DATE", "group"])[target_columns].mean().groupby("group").mean()
    # 计算每个group的样本数
    count_df = df_sz_temp.groupby("group").size().to_frame(name="count")
    # 合并样本数和单日平均收益为结果表
    result_df = pd.concat([count_df, rst_1], axis=1)
    # 调整列顺序（count在前，目标收益列在后）
    result_df = result_df[["count"] + target_columns]
    # 按group降序排列（9组在最前，0组在最后）
    result_df = result_df.sort_index()
    # 保存带高亮的Excel（最大值标红）
    result_df.style.highlight_max(
        subset=target_columns,
        axis=1
    ).to_excel(output_path, engine="openpyxl")  # 显式指定引擎避免错

# 获取当前文件夹下所有csv和xlsx文件
file_paths = [r"\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\tyx_ml\high深圳\test.feather"] # 测试集
ret_data = feather.read_dataframe(r"\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\开盘买入出场四收益.feather")
savepath = r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\tyx_ml\high深圳'
os.makedirs(savepath,exist_ok=True)
buy_every_top=True


for file_path in file_paths:
    # 获取文件名（不含路径）和扩展名
    file_name = os.path.basename(file_path)
    base_name = os.path.splitext(file_name)[0]  # 去掉扩展名的基名

    # 根据文件类型读取数据
    # try:
    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path,encoding='gbk')
        except:
            df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):  # xlsx文件
        df = pd.read_excel(file_path)
    elif file_path.endswith('.feather'):
        df = feather.read_dataframe(file_path)
    # except Exception as e:
    #     print(f"读取文件 {file_name} 出错: {e}")
    #     continue

    # 数据处理
    try:
        # 检查必要列是否存在
        df_cols = np.setdiff1d(df.columns,['TICKER','DATE'])[0]
        df = pd.merge(df,ret_data,how='left',on=['TICKER','DATE'])

        required_columns = [df_cols, 'DATE'] + target_columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        df['DATE'] = df['DATE'].astype(str)
        if missing_cols:
            print(f"文件 {file_name} 缺少必要列: {missing_cols}，跳过处理")
            continue

        # 裁剪收益列（限制极端值）并调整
        df[target_columns] = df[target_columns].clip(-0.4, 0.4)  # 限制在[-0.4, 0.4]
        df[target_columns] -= 0.0007  # 统一减去一个基准值

        # 计算排名和分组（按DATE分组计算score的百分比排名，再分为10组）
        if buy_every_top:
            df["rank"] = df.groupby("DATE")[df_cols].rank(pct=True)  # 0-1的百分比排名
        else:
            df["rank"] = df[df_cols].rank(pct=True)  # 0-1的百分比排名

        df["group"] = df["rank"].apply(lambda x:np.floor((x - 0.0000001) * 10) + 1)  # 转换为0-9的组

        # 筛选日期条件（只保留20250101之后的数据）
        df_filtered = df[df["DATE"] > '20250101']

        # 输出结果文件（当前文件夹，后缀_result）
        result_file = os.path.join(savepath, f"{base_name}_result.xlsx")
        pivot_adjusted(df_filtered, result_file)
        print(f"已生成结果文件: {result_file}")
        df_filtered_group = df_filtered.groupby('group')[target_columns].mean().reset_index(drop=False)
        # 输出底表文件（result文件夹，后缀_底表，仅保留0-9组且日期符合条件的数据）
        bottom_file = os.path.join(savepath, f"{base_name}_底表.xlsx")
        df_bottom = df[(df["group"].between(1, 10)) & (df["DATE"] > '20250101')]
        # print(df_bottom)
        df_bottom.to_excel(bottom_file, index=False, engine="openpyxl")
        print(f"已生成底表文件: {bottom_file}")

    except Exception as e:
        print(f"处理文件 {file_name} 时出错: {e}")
        continue

print("批量处理完成！")