from torch import nn
import torch.nn.functional as F

class MLP_my(nn.Module):
    def __init__(self,feature_nums,dim1,dim2,dim3,out_dim=1,drop_rate=0.5):
        super(MLP_my,self).__init__()
        self.dim1=dim1
        self.dim2=dim2
        self.out_dim=out_dim

        self.hidden1=nn.Linear(feature_nums,dim1,bias=True)
        self.dropout1 = nn.Dropout(drop_rate)
        self.hidden2=nn.Linear(dim1,dim2)
        self.dropout2 = nn.Dropout(drop_rate)
        self.hidden3=nn.Linear(dim2,dim3)
        self.dropout3 = nn.Dropout(drop_rate)
        self.out_layer=nn.Linear(dim3,out_dim)

    def forward(self,x):
        x=F.elu(self.hidden1(x))
        x=self.dropout1(x)
        x=F.elu(self.hidden2(x))
        x = self.dropout2(x)
        x=F.elu(self.hidden3(x))
        x = self.dropout3(x)
        out=self.out_layer(x)
        return out[:,0]

class EarlyStopping:
    def __init__(self, patience=3, min_delta=0):
        """
        参数：
        - patience: 忍耐轮数（如果验证集 loss 多轮不下降，就停止训练）
        - min_delta: 最小变化幅度（小于这个值就认为没有改进）
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss # 第一次调用时，初始化best_loss
        elif val_loss < self.best_loss - self.min_delta:  # 如果当前验证集loss比之前的best_loss降低超过了min_delta（即有明显改进）
            self.best_loss = val_loss
            self.counter = 0  # reset counter
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True  # # 如果连续多轮都没有提升，达到忍耐阈值，就设置early_stop为True