class DataPath():
    # 数据存储路径-----------------------------------------------------------------
    factor_out_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子'
    save_path_old=r'D:\tyx\检查更新\2020-2022.12'
    save_path_update=r'D:\tyx\检查更新\2021.6-2025.8'
    save_help_path=r'D:\tyx\raw_data'
    tmp_path=r'D:\tyx\中间数据'
    # tmp_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用'

    # 数据路径-------------------------------------------------------
    # daily_path=r'\\192.168.1.101\local_data\Data_Storage\daily.feather'
    daily_path=r'Z:\local_data\Data_Storage\daily.feather'
    # sh_min=r'\\DESKTOP-79NUE61\SH_min_data' # 上海数据 逐笔合分钟数据
    sh_min=r'\\192.168.1.210\SH_min_data' # 上海数据 逐笔合分钟数据
    sz_min=r'\\DESKTOP-79NUE61\SZ_min_data' # 深圳数据 逐笔合分钟数据
    feather_2022 = r'\\192.168.1.28\h\data_feather(2022)\data_feather'
    feather_2023 = r'\\192.168.1.28\h\data_feather(2023)\data_feather'
    feather_2024 = r'\\192.168.1.28\i\data_feather(2024)\data_feather'
    feather_2025 = r'\\192.168.1.28\i\data_feather(2025)\data_feather'
    moneyflow_sh=r'\\DESKTOP-79NUE61\money_flow_sh'
    moneyflow_sz=r'\\DESKTOP-79NUE61\money_flow_sz'
    # moneyflow数据按照4万，20万，100万为分界线，分small,medium,large,xlarge,此数据将集合竞价数据考虑进来了
    # mkt_index=r'\\192.168.1.101\local_data\ex_index_market_data\day'
    # to_df_path=r'\\192.168.1.101\local_data\Data_Storage' #float_mv.feather,adj_factors(复权因子)数据路径,citic_code.feather数据(行业数据),money_flow
    # to_data_path=r'\\192.168.1.101\local_data\base_data' #totalShares数据路径
    # to_path=r'\\192.168.1.101\local_data' #calendar.csv数据路径
    # wind_A_path=r'\\192.168.1.101\ssd\local_data\Data_Storage\881001.WI.csv' #万得全A指数路径
    # ind_df_path = r'\\192.168.1.101\local_data\Data_Storage\citic_code.feather'
    mkt_index = r'Z:\local_data\ex_index_market_data\day'
    to_df_path = r'Z:\local_data\Data_Storage'  # float_mv.feather,adj_factors(复权因子)数据路径,citic_code.feather数据(行业数据),money_flow
    to_data_path = r'Z:\local_data\base_data'  # totalShares数据路径
    to_path = r'Z:\local_data'  # calendar.csv数据路径
    wind_A_path = r'Z:\local_data\Data_Storage\881001.WI.csv'  # 万得全A指数路径
    ind_df_path = r'Z:\local_data\Data_Storage\citic_code.feather'

    # feather_sh=r'\\Desktop-79nue61\sh'
    # feather_sz=r'\\Desktop-79nue61\sz'
    feather_sh=r'\\192.168.1.210\sh'
    feather_sz=r'\\192.168.1.210\sz'

    order_min_sh=r'\\DESKTOP-79NUE61\SH_min_data_big_order'
    order_min_sz=r'\\DESKTOP-79NUE61\SZ_min_data_big_order'
    # 财务数据路径
    financial_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\财务数据'

    # -------------------------------------------
    # 机器学习数据路径
    train_data_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\rolling_generation_202508_sz_auction_end_20240815_908\train'
    train_big_order_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\rolling_generation_202508_sz_auction_end_20240815_909_big_order\train'

    # 最终因子路径
    factor_path=r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子'
    ret_df_path = r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\return_因子组合用.feather'

    snap_path_2022=fr'\\192.168.1.7\data01\data_feather'
    snap_path_2023=fr'\\192.168.1.7\data02\data_feather'
    snap_path_2024=fr'\\192.168.1.7\data03\data_feather'
    snap_path_2025=fr'\\192.168.1.7\data04\data_feather'
    # r'Z:\local_data\base_data\st.csv'去除st股票