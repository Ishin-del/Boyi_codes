import os
import warnings
import feather
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.manifold import TSNE
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
import joblib
from tool_funs import *
# 数据集：\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\吴文博alpha_pool\high上海.feather

def get_data(file_name):
    df=feather.read_dataframe(fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\吴文博alpha_pool\{file_name}.feather')
    df.rename(columns={'code':'TICKER','entry_date':'DATE'},inplace=True)
    if '平均卖出收益_1100_1400' not in df.columns:
        tmp=feather.read_dataframe(r'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\开盘买入出场四收益.feather')
        df=df.merge(tmp,on=['DATE','TICKER'],how='inner')
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.dropna(inplace=True)
    df.drop(columns=['exit_4_return','auction_amount'],inplace=True)
    df['平均卖出收益_1100_1400'] = df['平均卖出收益_1100_1400'].clip(-0.4, 0.4)  # 限制在[-0.4, 0.4]
    df['平均卖出收益_1100_1400'] -= 0.0007  # 统一减去一个基准值
    # pca---------------------------
    # df=normalize1(df)
    # pca=TSNE(n_components=3,random_state=0)
    # df=df[['TICKER','DATE','115','116','117','平均卖出收益_1100_1400']]
    # 数据分割-----------------------------------------------------------
    except_date=['20240206', '20240207', '20240208', '20240926', '20240927', '20240930', '20241008']
    special_df=df[df['DATE'].isin(except_date)]
    df=df[~df['DATE'].isin(except_date)]
    train_df=df[df['DATE']<='20240630']
    # print(train_df.describe())
    train_x,train_y=split_df(train_df,y_col='平均卖出收益_1100_1400')
    # pca.fit(train_x)
    # train_x=pca.fit_transform(train_x)
    valid_df=df[(df['DATE']>='20240701')&(df['DATE']<='20241231')]
    valid_x,vaild_y=split_df(valid_df,y_col='平均卖出收益_1100_1400')
    # valid_x=pca.fit_transform(valid_x)
    test_df=df[df['DATE']>='20250101']
    test_x,test_y=split_df(test_df,y_col='平均卖出收益_1100_1400')
    # test_x=pca.fit_transform(test_x)
    # -------------------
    # correlation_matrix = np.corrcoef(train_x.T)
    # if train_x.shape[1] < 20:
    #     import seaborn as sns
    #     sns.heatmap(correlation_matrix, annot=True)
    #     plt.title('特征相关性矩阵')
    #     plt.show()

    return train_x,train_y,valid_x,vaild_y,test_x,test_y


def ml_para():
    # model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=2, num_leaves=10, min_child_samples=5,
    #                           subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=12, verbose=-1)
    # model = XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    #                      random_state=42, n_jobs=12)
    # model=RandomForestRegressor(random_state=42,max_depth=3,n_estimators=100,min_samples_split=5,min_samples_leaf=3)
    # model=DecisionTreeRegressor(random_state=42,max_depth=4,min_samples_leaf=5,min_samples_split=10)
    # model=linear_model.ElasticNet(alpha=0.5, l1_ratio=0.5)#alpha=0.1, l1_ratio=0.5
    # model=linear_model.Lasso(alpha=50)#alpha=0.1, l1_ratio=0.5
    # model=linear_model.Ridge(alpha=50)
    model=linear_model.LinearRegression(positive=True)
    # param_grid = {
    #     # 'n_estimators': [100,150],
    #     # 'learning_rate': [0.01,0.005,0.0075],
    #     'max_depth': [3,4],
    #     # 'num_leaves': [5,6,7],
    #     # 'reg_alpha': [1,0.8],
    #     # 'reg_lambda': [2,1]
    #     'min_samples_split':[5,10,15],
    #     'min_samples_leaf':[5,10,15],
    #     # 'oob_score':[True,False]
    # }
    # grid_search = GridSearchCV(
    #     estimator=model,
    #     param_grid=param_grid,
    #     cv=5,
    #     scoring=ic_score,
    #     n_jobs=12)
    # return grid_search
    return model

def ml(file_name):
    path = fr'\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\tyx_ml\{file_name}'
    train_x,train_y,valid_x,valid_y,test_x,test_y=get_data(file_name)
    grid_search=ml_para()
    grid_search.fit(train_x, train_y)
    # print("最佳参数:", grid_search.best_params_)
    # print("最佳IC分数:", grid_search.best_score_)
    train_pred_y=grid_search.predict(train_x)
    train=pd.DataFrame(train_pred_y, index=train_y.index, columns=['pred'])
    # print(train)
    # group_mean(train)
    print('train ic:',pearsonr(train_pred_y.flatten(), train_y.values.flatten())[0])

    valid_pred_y = grid_search.predict(valid_x)
    valid = pd.DataFrame(valid_pred_y, index=valid_y.index, columns=['pred'])
    # group_mean(valid)
    # print(valid)

    print('valid ic:',pearsonr(valid_pred_y.flatten(), valid_y.values.flatten())[0])
    test_pred_y = grid_search.predict(test_x)
    test = pd.DataFrame(test_pred_y, index=test_y.index, columns=['pred'])

    # print(test)
    print('test ic:', pearsonr(test_pred_y.flatten(), test_y.values.flatten())[0])

    os.makedirs(path,exist_ok=True)
    group_mean(test, file_name)
    joblib.dump(grid_search, os.path.join(path,'linear.joblib'))
    feather.write_dataframe(train,os.path.join(path,'train.feather'))
    feather.write_dataframe(valid,os.path.join(path,'valid.feather'))
    feather.write_dataframe(test,os.path.join(path,'test.feather'))
    print('保存成功')



if __name__=='__main__':
    ml('成交量增加上海')