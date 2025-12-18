'''
计算收益率，考虑是否adj，用close计算或者用分钟数据计算区别是？准确到第几分钟买入吗 ==> 看盘，成交量在早上与下午的分别，考虑会对交易造成的影响
----------------------------------
相关性小
捕捉趋势：用价格动量因子，中期动量（20天） ==> 如果这么写，赚的就应该是这部分的钱
均值回归：用短期反转（1天到5天）
成交量异常：关注资金流向
波动率：20日波动率

逻辑大概是：用动量因子取捕捉股票趋势，但要用反转因子去考虑均值回归的情况，波动率是要考虑风险，同时关注成交量异常情况
'''
import warnings
import feather
import os
import numpy as np
import pandas as pd
from dask.array import shape
from scipy.stats import pearsonr
from sklearn import linear_model
warnings.filterwarnings('ignore')
'''
先检测因子间相关性以及ic,进行因子选择：(初步选择)
1. 捕捉趋势: reduce_fell_roll20
2. 波动率: Parkinson
3. 反转+动量: Amn_diff_MeanRisk_factor_adjust
 ---------------------------------
 先尝试线性模型
'''
class SS:
    def __init__(self, start=None, end=None, calender_path=None, factor_path=None, daily_path=None,st_path=None,
                 train_ratio=0.7,valid_ratio=0.15):
        self.start = start or '20220101'
        self.end = end or '20241231'
        self.factor_path=r'D:\tyx\数据\实验数据集' or factor_path
        self.daily_path = daily_path or r'D:\tyx\数据\实验数据集\daily.feather'
        self.train_ratio=train_ratio
        self.vaild_ratio=valid_ratio
        self.calender_path = calender_path or r'D:\tyx\数据\实验数据集\calendar.csv'
        # y数据---------------------------------------------
        self.daily_df = feather.read_dataframe(self.daily_path)[['DATE', 'TICKER', 'close']]
        self.daily_df.sort_values(['TICKER', 'DATE'], inplace=True)
        self.daily_df['close']=self.daily_df.groupby('TICKER')['close'].shift(1)
        self.daily_df['daily_ret'] = self.daily_df.groupby('TICKER')['close'].pct_change()
        self.daily_df=self.daily_df[['DATE', 'TICKER', 'daily_ret']]
        self.tar_label='daily_ret'  # y列的名字
        # 交易日数据----------------------------------------
        calendar = pd.read_csv(self.calender_path, dtype={'trade_date': str})
        calendar.rename(columns={'trade_date': 'DATE'}, inplace=True)
        self.daily_list = calendar[(calendar['DATE'] >= self.start) & (calendar['DATE'] <= self.end)]['DATE']
        # ------------------------------------------------
        # 获取train,valid,test对应的日期
        valid_ratio += train_ratio
        self.train_list = self.daily_list[:int(len(self.daily_list) * train_ratio)]
        self.valid_list = self.daily_list[int(len(self.daily_list) * train_ratio):int(len(self.daily_list) * valid_ratio)]
        self.test_list = self.daily_list[int(len(self.daily_list) * valid_ratio):]
        # ----------------------------------------------------
        self.st_path=st_path or r'D:\tyx\数据\实验数据集\st.csv'
        self.st_df=pd.read_csv(self.st_path,encoding='gbk')
        self.st_df.rename(columns={'Unnamed: 0':'DATE'},inplace=True)
        self.st_df['DATE']=self.st_df['DATE'].str.replace('-','')
        self.st_df.set_index('DATE',inplace=True)

    def get_df(self):
        def scale_x(x):
            return (x - x.mean()) / x.std()
        main_df=feather.read_dataframe(os.path.join(self.factor_path,'reduce_fell_roll20.feather'))
        # 照这个逻辑 赚的应该是日内成交额下跌,收益趋势下跌,但在长期却会反转的,这部分钱
        vol_df=feather.read_dataframe(os.path.join(self.factor_path,'Parkinson.feather')) #根据公式捕捉的波动率因子
        amn_df=feather.read_dataframe(os.path.join(self.factor_path,'Amn_diff_MeanRisk_factor_adjust.feather'))
        #以成交额变动均值作为风险权重
        df=main_df.merge(vol_df,on=['DATE','TICKER'],how='inner').merge(amn_df,on=['DATE','TICKER'],how='inner')
        # =============
        df=df.merge(self.daily_df,on=['DATE','TICKER'],how='inner')
        # 剔除st股票
        self.st_df=self.st_df[self.st_df.columns[self.st_df.columns.isin(df['TICKER'].drop_duplicates())]].reset_index()
        self.st_df=self.st_df.melt(id_vars=['DATE'],var_name='TICKER',value_name='st')
        df=df.merge(self.st_df,on=['DATE','TICKER'],how='inner')
        df=df[df['st']=='否'].drop(columns='st')
        df.replace([np.inf,-np.inf],np.nan,inplace=True)
        df.dropna(inplace=True)
        # -------------------------------------
        # 数据中性化
        df.set_index(['TICKER'], inplace=True)
        df = df.groupby('DATE').apply(scale_x).reset_index()
        # -------------------------------------
        # 数据分割
        train_df = df[df['DATE'].isin(self.train_list)].set_index(['TICKER','DATE'])
        valid_df = df[df['DATE'].isin(self.valid_list)].set_index(['TICKER','DATE'])
        test_df= df[df['DATE'].isin(self.test_list)].set_index(['TICKER','DATE'])
        chars=np.setdiff1d(train_df.columns,self.tar_label)
        train_x,train_y=train_df[chars],train_df[[self.tar_label]]
        valid_x,valid_y=valid_df[chars],valid_df[[self.tar_label]]
        test_x,test_y=test_df[chars],test_df[[self.tar_label]]
        return train_x,valid_x,test_x ,train_y,valid_y,test_y


    def sss(self):
        train_x, valid_x, test_x, train_y, valid_y, test_y=self.get_df()
        model = linear_model.LinearRegression()
        model.fit(train_x, train_y)
        train_pred_y=model.predict(train_x)
        train = pd.DataFrame(train_pred_y, index=train_y.index, columns=['pred'])
        train_res=self.group_ret(train)
        train_res['IC']=pearsonr(train_pred_y.flatten(), train_y.values.flatten())[0]
        train_res['start_time']=train_y.reset_index(level='DATE')['DATE'].min()
        train_res['end_time']=train_y.reset_index(level='DATE')['DATE'].max()
        # train_res['数据集']='训练集'
        # ========================================================================
        valid_pred_y = model.predict(valid_x)
        valid = pd.DataFrame(valid_pred_y, index=valid_y.index, columns=['pred'])
        valid_res=self.group_ret(valid)
        valid_res['IC']=pearsonr(valid_pred_y.flatten(), valid_y.values.flatten())[0]
        valid_res['start_time']=valid_y.reset_index(level='DATE')['DATE'].min()
        valid_res['end_time']=valid_y.reset_index(level='DATE')['DATE'].max()
        # valid_res['数据集']='验证集'
        # ========================================================================
        test_pred_y = model.predict(test_x)
        test = pd.DataFrame(test_pred_y, index=test_y.index, columns=['pred'])
        test_res=self.group_ret(test)
        test_res['IC']=pearsonr(test_pred_y.flatten(), test_y.values.flatten())[0]
        test_res['start_time']=test_y.reset_index(level='DATE')['DATE'].min()
        test_res['end_time']=test_y.reset_index(level='DATE')['DATE'].max()
        # test_res['数据集']='测试集'
        res_df=pd.concat([train_res,valid_res,test_res]).reset_index(drop=True).T
        res_df.columns=['训练集', '验证集', '测试集']
        print(res_df)

    def group_ret(self,df):
        df=df.merge(self.daily_df,on=['DATE','TICKER'],how='inner')
        df['rank'] = df.groupby('DATE')['pred'].rank(pct=True)  # 0-1的百分比排名
        df['rank'] = df['pred'].rank(pct=True)  # 0-1的百分比排名
        df['group'] = df['rank'].apply(lambda x: np.floor((x - 0.0000001) * 10) + 1)
        # np.floor向比自己更小的数取整
        # df.sort_values('pred',inplace=True)
        # df['group'] = pd.cut(df.index, bins=10, labels=False)+1
        df=df.groupby(['DATE','group'])[self.tar_label].mean().reset_index()
        # df=df.groupby('group')[self.tar_label].mean().reset_index()
        # print(df)
        # ------------------------
        ret_df=df[df['group']==10].set_index('DATE')[self.tar_label]
        annual_ret=ret_df.mean()*252 #年化收益
        annual_std=ret_df.std()*np.sqrt(252) #年化波动
        shape_ratio=annual_ret/annual_std #夏普比率
        cum_ret=(1+ret_df).prod()-1
        cumulative=(1+ret_df).cumprod() #累计收益
        # 最大回撤
        draw_down=(cumulative/cumulative.expanding().max()-1).min()
        win_rate=(ret_df>0).mean()
        res={'累计收益':cum_ret,'年化收益':annual_ret,'年化波动':annual_std,'夏普比率':shape_ratio,'最大回撤':draw_down,'胜率':win_rate}
        return pd.DataFrame([res])

'''
todo:
策略多久触发一次（schedule）
上市不足100日的新股和退市股和B股
# ---------------------------
账户仓位情况：
如果预测值达到标准，则开仓
卖出不在标的池的仓位
（注意仓位止盈，止损）
'''

if __name__=='__main__':
    obj=SS(start='20220101', end='20251031')
    obj.sss()

