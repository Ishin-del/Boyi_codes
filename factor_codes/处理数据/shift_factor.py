import feather
from tqdm import tqdm

tar_ll=['Amn_diff_MeanRisk_factor_adjust','Amn_diff_StdRisk_factor_adjust',
'BuyAmn_diff_MeanRisk_factor_adjust','buy_concentration','cVaR_RT_roll5',
'ES_1%','ideal_v_diff','ideal_v_low','min_ret_risk_factor_adjust',
'MSF_half10','retOnBuyDiff_bottom_24_roll20','retOnBuyDiff_top_24_roll20',
'retOnNetAct_top_24_roll20','risk_1_factor_adjust','RV_roll5','sell_midd_ratio',
'sell_trade_ratio','SSF_half10','VaR_5%_v2','一视同仁因子','有效100s买入额',
'波动公平因子_roll20','随波逐流']
df=feather.read_dataframe(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\20251030拼因子用return.feather')
for fac in tqdm(tar_ll):
    tmp=feather.read_dataframe(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\原始数据\拟使用量价因子\{fac}.feather')
    tmp.sort_values(['TICKER','DATE'],inplace=True)
    tmp[fac]=tmp.groupby('TICKER')[fac].shift(1)
    df=df.merge(tmp,on=['DATE','TICKER'],how='inner')

feather.write_dataframe(df,r'C:\Users\admin\Desktop\fac_ret.feather')