from sklearn.preprocessing import StandardScaler
import torch.utils.data as Data
from tool_funs import *
from tyx_tttt.DeepModel import MLP_my,EarlyStopping
import torch
import torch.nn as nn
# 数据集：\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\吴文博alpha_pool\low上海.feather

def get_data(file_name,time_stop):
    df=feather.read_dataframe(fr'\\192.168.1.210\Factor_Storage\田逸心\calc6临时用\{file_name}.feather') #吴文博alpha_pool\
    if '平均卖出收益_1100_1400' not in df.columns:
        tmp=feather.read_dataframe(r'\\192.168.1.210\Factor_Storage\田逸心\calc6临时用\开盘买入出场四收益.feather')
        df=df.merge(tmp,on=['DATE','TICKER'],how='inner')
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.replace(np.nan,0,inplace=True)
    df['平均卖出收益_1100_1400'] = df['平均卖出收益_1100_1400'].clip(-0.4, 0.4)  # 限制在[-0.4, 0.4]
    df['平均卖出收益_1100_1400'] -= 0.0007  # 统一减去一个基准值
    # 数据分割-----------------------------------------------------------
    except_date=['20240206', '20240207', '20240208', '20240926', '20240927', '20240930', '20241008']
    special_df=df[df['DATE'].isin(except_date)]
    df=df[~df['DATE'].isin(except_date)]
    train_df=df[df['DATE']<=time_stop[0]]
    valid_df = df[(df['DATE'] >= time_stop[1]) & (df['DATE'] <= time_stop[2])]
    test_df = df[df['DATE'] >= time_stop[3]]
    # -------------------------------------------------------------
    train_x,train_y=split_df(train_df,y_col='平均卖出收益_1100_1400')
    valid_x,valid_y=split_df(valid_df,y_col='平均卖出收益_1100_1400')
    test_x,test_y=split_df(test_df,y_col='平均卖出收益_1100_1400')
    return train_x,train_y,valid_x,valid_y,test_x,test_y,train_y.index,valid_y.index,test_y.index


def ml_para(flag='label1'):
    model=MLP_my(97,64,32,16,1)
    return model

def ml(file_name,time_stop=['20240630','20240701','20241231','20250101'],flag='label1'):
    torch.manual_seed(42) #3407
    path = fr'\\192.168.1.210\Factor_Storage\田逸心\calc6临时用\tyx_ml\{file_name}'
    train_x, train_y, valid_x, valid_y, test_x, test_y,train_y_index,valid_y_index,test_y_index = get_data(file_name, time_stop)
    # 数据转换-----------------------
    scale = StandardScaler()
    train_x = scale.fit_transform(train_x)
    valid_x = scale.transform(valid_x)
    test_x = scale.transform(test_x)

    train_x = torch.from_numpy(train_x.astype(np.float32))
    train_y = torch.from_numpy(train_y.values.astype(np.float32).flatten())
    valid_x = torch.from_numpy(valid_x.astype(np.float32))
    valid_y = torch.from_numpy(valid_y.values.astype(np.float32).flatten())
    test_x = torch.from_numpy(test_x.astype(np.float32))
    test_y = torch.from_numpy(test_y.values.astype(np.float32).flatten())

    train_data = Data.TensorDataset(train_x, train_y)
    valid_data = Data.TensorDataset(valid_x, valid_y)
    test_data = Data.TensorDataset(test_x, test_y)
    train_loader = Data.DataLoader(dataset=train_data, batch_size=64,shuffle=True, num_workers=0,drop_last=False)
    valid_loader = Data.DataLoader(dataset=valid_data, batch_size=64,shuffle=False, num_workers=0,drop_last=False)
    # -----------------------------------------
    grid_search = ml_para(flag)
    #  dl参数设置--------------------------------
    optimizer=torch.optim.Adam(grid_search.parameters(), lr = 0.001)
    loss_func=nn.MSELoss()
    early_stopping = EarlyStopping(patience=20)
    # train_loss_all=[]
    for epoch in range(100):
        grid_search.train() #训练
        train_loss,train_num=0,0
        for step, (b_x, b_y) in enumerate(train_loader):
            train_pred_y = grid_search(b_x)  # MLP在训练batch上的输出
            loss = loss_func(train_pred_y, b_y)  # 均方根损失函数
            optimizer.zero_grad()  # 每次迭代梯度初始化0
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 使用梯度进行优化
            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
        train_loss=train_loss / train_num
        # -----------------------------------
        valid_loss,valid_num=0,0
        grid_search.eval() #验证
        with torch.no_grad():
            for step,(b_x,b_y) in enumerate(valid_loader):
                valid_pred_y=grid_search(b_x)
                loss=loss_func(valid_pred_y,b_y)
                valid_loss += loss.item() * b_x.size(0)
                valid_num += b_x.size(0)
            valid_loss=valid_loss/valid_num
            early_stopping(valid_loss)
        print(f'epoch:{epoch}',f'train avg loss:{train_loss}',f'valid avg loss:{valid_loss}')
        if early_stopping.early_stop:
            print("Early stopping!")
            break


    train_pred_y = grid_search(train_x).data.numpy()
    valid_pred_y = grid_search(valid_x).data.numpy()
    test_pred_y = grid_search(test_x).data.numpy()

    train=pd.DataFrame(train_pred_y, index=train_y_index, columns=['pred'])
    valid=pd.DataFrame(valid_pred_y, index=valid_y_index, columns=['pred'])
    test=pd.DataFrame(test_pred_y, index=test_y_index, columns=['pred'])
    # ------------------------------------------------------
    group_mean(train)
    print('train ic:', pearsonr(train_pred_y.flatten(), train_y.flatten())[0])
    group_mean(valid)
    print('valid ic:', pearsonr(valid_pred_y.flatten(), valid_y.flatten())[0])
    group_mean(test)
    print('test ic:', pearsonr(test_pred_y.flatten(), test_y.flatten())[0])

def group_mean(aa,file_name=''):
    ret=feather.read_dataframe(r'C:\Users\admin\Desktop\ret.feather')
    # ret.loc[ret['平均卖出收益_1100_1400'] > 0.4, '平均卖出收益_1100_1400'] = 0.4
    ret['平均卖出收益_1100_1400'] = ret['平均卖出收益_1100_1400'].clip(-0.4, 0.4)  # 限制在[-0.4, 0.4]
    ret['平均卖出收益_1100_1400'] -= 0.0007  # 统一减去一个基准值
    ret['DATE'] = ret['DATE'].astype(str)
    ret.rename(columns={'平均卖出收益_1100_1400':'ret'},inplace=True)
    zz = pd.merge(ret, aa, how='inner', on=['TICKER', 'DATE'])
    usename = list(np.setdiff1d(aa.columns, ['TICKER', 'DATE']))[0]
    # bins = pd.qcut(zz[usename], 10)
    zz['group'] = pd.qcut(zz[usename], 10, labels=False, duplicates='drop') + 1
    # res = zz.groupby('group')['ret'].mean().reset_index(drop=False)
    res=zz.groupby(['DATE', 'group'])['ret'].mean().reset_index(drop=False).groupby('group')['ret'].mean()
    print(res)
    # res.to_csv(fr'\\192.168.1.210\Factor_Storage\田逸心\calc6临时用\tyx_ml\{file_name}\test_res.csv',index=False)
    return res


if __name__=='__main__':
    # ml('20251115_sz_auction_processed')
    ml('20251115_sz_auction_processed',['20220630','20220701','20221231','20230101'],'label2')
    # ml('20251115_sz_auction_processed',['20221231','20230101','20230630','20230701'],'label3')
    # print('-------------------')
    # ll=[['20220630','20220701','20221231','20230101'],['20221231','20230101','20230630','20230701']]
    # n=2
    # for l in ll:
    #     ml('20251115_sz_auction_processed',l,f'label{n}')
    #     n+=1
    #     print('----------------------')


