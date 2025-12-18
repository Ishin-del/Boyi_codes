import warnings
import torch
import torch.nn as nn
import feather
import numpy as np
import pandas as pd
import torch.optim as optim
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 数据集：C:\Users\admin\Desktop\high深圳.feather

def get_data():
    df=feather.read_dataframe(r'C:\Users\admin\Desktop\high深圳.feather')
    df.drop(columns=['exit_4_return','auction_amount','开盘后10秒vwap收益_1100_1400','开盘后30秒vwap收益_1100_1400',
                     '开盘后60秒vwap收益_1100_1400','开盘后180秒vwap收益_1100_1400','开盘后300秒vwap收益_1100_1400',
                     '开盘后600秒vwap收益_1100_1400','开盘后1200秒vwap收益_1100_1400','开盘后1800秒vwap收益_1100_1400'],
            inplace=True)

    except_date=['20240206', '20240207', '20240208', '20240926', '20240927', '20240930', '20241008']
    df.dropna(inplace=True)
    special_df=df[df['entry_date'].isin(except_date)]
    df=df[~df['entry_date'].isin(except_date)]
    train_df=df[df['entry_date']<='20240630']
    train_x,train_y=split_df(train_df)
    valid_df=df[(df['entry_date']>='20240701')&(df['entry_date']<='20241231')]
    valid_x,vaild_y=split_df(valid_df)
    test_df=df[df['entry_date']>='20250101']
    test_x,test_y=split_df(test_df)
    return train_x,train_y,valid_x,vaild_y,test_x,test_y

def split_df(df):
    warnings.filterwarnings('ignore')
    df.rename(columns={'code':'TICKER','entry_date':'DATE'},inplace=True)
    df.set_index(['TICKER','DATE'],inplace=True)
    y=df[['平均卖出收益_1100_1400']]
    x=df.drop(columns='平均卖出收益_1100_1400')
    return x,y


class SMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SMLP, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)



def train_model(model, train_x, train_y, valid_x, valid_y, epochs=300, lr=0.01):
    x_tensor = torch.tensor(train_x.values, dtype=torch.float32)
    y_tensor = torch.tensor(train_y.values, dtype=torch.float32)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        outputs = model(x_tensor)
        loss = criterion(outputs, y_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.6f}')

    return model


def ml():
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_data()
    model = SMLP(train_x.shape[1], 1)

    model = train_model(model, train_x, train_y, valid_x, valid_y, epochs=100, lr=0.001)

    train_pred_y = model(train_x)
    train_df = pd.DataFrame(train_pred_y, index=train_y.index, columns=['pred'])
    group_mean(train_df)
    print('train ic:', np.corrcoef(train_pred_y.flatten(), train_y.values.flatten())[0, 1])

    valid_pred_y = model(valid_x)
    valid_df = pd.DataFrame(valid_pred_y, index=valid_y.index, columns=['pred'])
    group_mean(valid_df)
    print('valid ic:', np.corrcoef(valid_pred_y.flatten(), valid_y.values.flatten())[0, 1])

    test_pred_y = model(test_x)
    test_df = pd.DataFrame(test_pred_y, index=test_y.index, columns=['pred'])
    group_mean(test_df)
    print('test ic:', np.corrcoef(test_pred_y.flatten(), test_y.values.flatten())[0, 1])

def group_mean(aa):
    ret=feather.read_dataframe(r'C:\Users\admin\Desktop\ret.feather')
    ret.loc[ret['平均卖出收益_1100_1400'] > 0.4, '平均卖出收益_1100_1400'] = 0.4
    ret['DATE'] = ret['DATE'].astype(str)
    ret.rename(columns={'平均卖出收益_1100_1400':'ret'},inplace=True)
    zz = pd.merge(ret, aa, how='inner', on=['TICKER', 'DATE'])
    usename = list(np.setdiff1d(aa.columns, ['TICKER', 'DATE']))[0]
    # bins = pd.qcut(zz[usename], 10)
    zz['group'] = pd.qcut(zz[usename], 10, labels=False, duplicates='drop') + 1
    res = zz.groupby('group')['ret'].mean().reset_index(drop=False)
    print(res)
    return res


if __name__=='__main__':
    ml()