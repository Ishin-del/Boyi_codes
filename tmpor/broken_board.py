import numpy as np
import pandas as pd

def get_target_stock():
    df = pd.read_feather(r'\\zkh\O\Data_Storage\Data_Storage\daily.feather')[['DATE', 'TICKER', 'close','high','low',
                                                                              'max_up','max_down']]
    df['broken_board']= np.isclose(df.high,df.max_up) & ~ np.isclose(df.close,df.max_up)
    df['fall_board']= np.isclose(df.low,df.max_down) & ~ np.isclose(df.close,df.max_down)
    df=df[df.DATE>='20210101']
    broken_board=df[df.broken_board][['DATE','TICKER']]
    fall_board=df[df.fall_board][['DATE','TICKER']]
    broken_board.to_feather(r'C:\Users\Administrator\Desktop\broken_board.feather')
    fall_board.to_feather(r'C:\Users\Administrator\Desktop\fall_board.feather')

if __name__=='__main__':
    get_target_stock()