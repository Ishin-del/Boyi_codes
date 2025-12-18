import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import pandas as pd
import os
import feather

def crawler_wind(url='https://wx.wind.com.cn/indexofficialwebsite/Kline?indexId'):
    response=requests.get(url)
    soup = BeautifulSoup(response.text, 'lxml')
    json_str=str(soup).replace('</p></body></html>','').replace('<html><body><p>','')
    data = json.loads(json_str)
    df = pd.DataFrame(data['Result'])[['tradeDate','open','close','volume','amount']]
    df['DATE']=pd.to_datetime(df['tradeDate'], unit='ms').dt.strftime('%Y%m%d')
    df.drop(columns='tradeDate',inplace=True)
    df=df[['DATE','open','close','volume','amount']]
    return df

def update(path=None):
    data=crawler_wind()
    if os.path.exists(os.path.join(path,'881001.WI.csv')):
        old=feather.read_dataframe(os.path.join(path,'881001.WI.csv'))
        test=old.merge(data,how='inner',on='DATE').tail(20)
        gap_len=len(old.columns)-1
        for i in range(len(old.columns)):
            if not np.isclose(test.iloc[:,1+i],test.iloc[:,1+gap_len+i]):
                print('数据出错！')
                exit()
        data=data[data.DATE>old.DATE.max()]
        old=pd.concat([old,data]).reset_index(drop=True)
        # feather.write_dataframe(old,os.path.join(path,'881001.WI.csv'))
        print(old)
    else:
        # feather.write_dataframe(data,os.path.join(path,'881001.WI.csv'))
        pass


if __name__=='__main__':
    update(path=r'C:\Users\Administrator\Desktop')
