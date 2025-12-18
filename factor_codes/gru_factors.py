# @Author: Yixin Tian
# @File: ml_factor.py
# @Date: 2025/9/8 10:21
# @Software: PyCharm
import os
import feather
import numpy as np
import pandas as pd
from joblib import Parallel,delayed
from torch.utils.data import TensorDataset
from tqdm import tqdm
from tool_tyx.path_data import DataPath
import torch.nn as nn
import torch
from torch.autograd import Variable
class Config():
    batch_size =20
    feature_size=69
    hidden_size=64
    output_size=1
    num_layers=1
    epochs=50
    best_loss=0
    learning_rate=0.001
    model_name='gru'

# def split_data(data,feature_size):

class GRU(nn.Module):
    def __init__(self,feature_size,hidden_size,num_layers,output_size):
        # feature_size:特征数量，69
        super(GRU,self).__init__()
        self.hidden_size=hidden_size # 隐层大小
        self.num_layers=num_layers #gru层数
        self.gru=nn.GRU(feature_size,hidden_size,num_layers,batch_first=True)
        self.fc=nn.Linear(hidden_size,output_size)
    def forward(self,x,hidden=None):
        batch_size=x.shape[0] # 获得行数，批次大小
        if hidden is None:
            h_0=x.data.new(self.num_layers,batch_size,self.hidden_size).fill_(0).float()
        else:
            h_0=hidden
        output,h_0=self.gru(x,h_0) #gru运算
        batch_size,timestep,hidden_size=output.shape #行数,时间片,列数
        output=output.reshape(-1,hidden_size) #将数据变成batch_size*timestep,hidden_dim
        output=self.fc(output)
        output=output.reshape(timestep,batch_size,-1)
        return output[-1] # 只需返回最后一个时间片的数据

def loss_corr(y_pred,y):
    mean_y=torch.mean(y)
    mean_y_pred=torch.mean(y_pred)
    cov=torch.mean((y-mean_y)*(y_pred-mean_y_pred))
    std_y=torch.std(y)
    std_y_pred=torch.std(y_pred)
    corr=cov/(std_y*std_y_pred)
    return corr

class CorrLoss(nn.Module):
    def __init__(self):
        super(CorrLoss,self).__init__()
    def forward(self,pred,label):
        pred_var=Variable(pred,requires_grad=True)
        label_var=Variable(label,requires_grad=True)
        corr=loss_corr(pred_var,label_var)
        loss=1-corr
        return loss

def read_data(date):
    x=feather.read_dataframe(os.path.join(DataPath.train_big_order_path,f'{date}_train_x.feather'))
    x.replace([np.inf,-np.inf],np.nan,inplace=True)
    y=feather.read_dataframe(os.path.join(DataPath.train_big_order_path,f'{date}_train_y_ori.feather'))
    y.replace([np.inf,-np.inf],np.nan,inplace=True)
    return [x,y]

def group_mean(true_data:pd.Series,pre_data:np.array,num):
    df = pd.DataFrame(true_data).reset_index(drop=True).merge(pd.DataFrame(pre_data), left_index=True,
                right_index=True).rename(columns={'ret': 'true_value', 0: 'pre_value'})
    df.sort_values(['pre_value'], ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['group'] = df.index // num
    res = df.groupby('group')['true_value'].mean()
    return res

def run_model(train_loader,test_loader):
    config=Config()
    model=GRU(config.feature_size,config.hidden_size,config.num_layers,config.output_size)
    loss_function=CorrLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate)
    for epoch in range(config.epochs):
        model.train()
        running_loss=0
        train_bar=tqdm(train_loader)
        for data in train_bar:
            x_train,y_train=data
            optimizer.zero_grad()# 梯度清0
            y_train_pred=model(x_train)
            loss=loss_function(y_train_pred,y_train.reshap(-1,1))
            loss.backward()
            optimizer.step()
            running_loss+=loss.items()
            print(f'train epoch [{epoch+1}/{config.epochs}] loss: {loss}')
    # todo:模型验证
    model.eval()
    test_loss=0
    with torch.no_grad():
        test_bar=tqdm(test_loader)
        for data in test_bar:
            x_test,y_test=data
            y_test_pred=model(x_test)
            test_loss =loss_function(y_test_pred,y_test.reshap(-1,1))

def run():
    tar_list = sorted(list(set([x[:8] for x in os.listdir(DataPath.train_big_order_path)])))
    # test_list = tar_list[-120:]
    # train_list = sorted(list(set(tar_list).difference(set(test_list))))
    train_list=[x for x in tar_list if x>='20220104' and x<='20240201']
    test_list=[x for x in tar_list if x>='20240202' and x<='20240731' and x not in ['20240206', '20240207', '20240208']]
    train_data_list=Parallel(n_jobs=15)(delayed(read_data)(date) for date in tqdm(train_list,desc='train datas are preparing'))
    train_data_list1=pd.concat([x[0] for x in train_data_list])
    train_data_list2=pd.concat([x[1] for x in train_data_list])
    train_data_list=[train_data_list1,train_data_list2]
    test_data_list = Parallel(n_jobs=15)(delayed(read_data)(date) for date in tqdm(test_list,desc='test datas are preparing'))
    test_data_list1 = pd.concat([x[0] for x in test_data_list])
    test_data_list2 = pd.concat([x[1] for x in test_data_list])
    test_data_list=[test_data_list1,test_data_list2]
    # ----------------------------------------
    config=Config()
    # ----------------------------------------
    train_data=TensorDataset(train_data_list1,train_data_list2)
    test_data=TensorDataset(test_data_list1,test_data_list2)
    train_loader=torch.utils.data.DataLoader(train_data,config.batch_size,False)
    test_loader=torch.utils.data.DataLoader(test_data,config.batch_size,False)
    run_model(train_loader, test_loader)


if __name__=='__main__':
    run()