import os
import feather
import pandas as pd
from tqdm import tqdm
import warnings
import sys
sys.path.append('../')
warnings.filterwarnings(action='ignore')

# 招商证券 高频寻踪：再觅知情交易者的踪迹
# VCV因子

class VCV(object):
    def __init__(self, start=None, end=None, path=None,test_path=None,data = pd.DataFrame(),stock_pool=None,savepath=None):
        self.data = data
        self.testpath = test_path or r'\\192.168.1.210\Factor_Storage\田逸心\原始数据\拟使用量价因子'  # 个人因子库路径
        self.start = start or '20240124'   # 开始日期
        self.end = end or '20240131'   # 结束日期
        # todo:
        self.stock_pool=stock_pool or r'\\zkh\O\Data_Storage\Data_Storage\daily.feather'
        self.path = path or r'\\zkh\O\Data_Storage\Data_Storage'   # 基础数据文件夹路径，一般是datacenter SSD盘Data_Storage
        self.savepath = savepath or r'C:\Users\Administrator\Desktop'   # 基个人备份保存路径

        self.sh_path=r'\\zkh\O\Data_Storage\tyx_min'
        self.sz_path=r'\\zkh\O\Data_Storage\tyx_min'

    def __load_data(self):
        res=[]
        use_ticker = self.stock_pool.TICKER.drop_duplicates()
        use_date = self.stock_pool.DATE.drop_duplicates()
        for path in [self.sh_path,self.sz_path]:
            for file in tqdm(os.listdir(path)):
                if file[:8] in list(use_date):
                    tmp=feather.read_dataframe(os.path.join(path,file))
                    tmp=tmp[tmp.TICKER.isin(use_ticker)]
                    res.append(tmp)
            break
        res=pd.concat(res).reset_index(drop=True)[['TICKER','DATE','min','volume']]
        return res

    def cal(self):
        data = feather.read_dataframe(os.path.join(self.path ,'daily.feather'),
                                      columns=['TICKER', 'DATE', 'volume'])
        data = data[(data['DATE'] <= self.end) & (data['DATE'] >= self.start)]
        # todo:
        data=data[data.TICKER.str.endswith('.SH')]
        self.stock_pool=data.copy()

        df = self.__load_data()
        # 日频因子计算
        df=df[df['min']!=930]
        df = df[(df['min'] < 1457) | (df['min'] > 1459)]
        df3=df.groupby(['TICKER','DATE'])['volume'].agg(['std','mean']).reset_index()
        df3['vcv_daily']=df3['std']/df3['mean']
        df3.drop(columns=['std', 'mean'],inplace=True)
        df3=df3[['TICKER', 'DATE','vcv_daily']]
        df3.to_feather(os.path.join(os.path.join(self.savepath,'check_daily.feather')))

    def run(self):
        self.cal()


if __name__ == '__main__':
    object = VCV()
    # break_point = 1
    object.run()
