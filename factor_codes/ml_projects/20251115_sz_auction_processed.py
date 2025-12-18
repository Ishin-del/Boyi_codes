from xgboost import XGBRegressor
from tool_funs import *


# 数据集：\\DESKTOP-79NUE61\Factor_Storage\田逸心\calc6临时用\吴文博alpha_pool\low上海.feather

def get_data(file_name,time_stop):
    df=feather.read_dataframe(fr'\\192.168.1.210\Factor_Storage\田逸心\calc6临时用\{file_name}.feather') #吴文博alpha_pool\
    # df.rename(columns={'code':'TICKER','entry_date':'DATE'},inplace=True)
    if '平均卖出收益_1100_1400' not in df.columns:
        tmp=feather.read_dataframe(r'\\192.168.1.210\Factor_Storage\田逸心\calc6临时用\开盘买入出场四收益.feather')
        df=df.merge(tmp,on=['DATE','TICKER'],how='inner')
    df.replace([np.inf,-np.inf],np.nan,inplace=True)
    df.replace(np.nan,0,inplace=True)
    # df.drop(columns=['exit_4_return','auction_amount'],inplace=True)
    # df.drop(columns=['开盘后10秒vwap收益_1100_1400','开盘后30秒vwap收益_1100_1400','开盘后60秒vwap收益_1100_1400','开盘后180秒vwap收益_1100_1400','开盘后300秒vwap收益_1100_1400','开盘后600秒vwap收益_1100_1400','开盘后1200秒vwap收益_1100_1400','开盘后1800秒vwap收益_1100_1400'],inplace=True)
    # df.dropna(inplace=True, axis=1)
    df['平均卖出收益_1100_1400'] = df['平均卖出收益_1100_1400'].clip(-0.4, 0.4)  # 限制在[-0.4, 0.4]
    df['平均卖出收益_1100_1400'] -= 0.0007  # 统一减去一个基准值
    # pca---------------------------
    # df=normalize(df)
    # df=normalize1(df)
    # pca=TSNE(n_components=3,random_state=0)
    # df=df[['TICKER','DATE','115','116','117','平均卖出收益_1100_1400']]
    # 数据分割-----------------------------------------------------------
    except_date=['20240206', '20240207', '20240208', '20240926', '20240927', '20240930', '20241008']
    special_df=df[df['DATE'].isin(except_date)]
    df=df[~df['DATE'].isin(except_date)]
    train_df=df[df['DATE']<=time_stop[0]]
    valid_df = df[(df['DATE'] >= time_stop[1]) & (df['DATE'] <= time_stop[2])]
    test_df = df[df['DATE'] >= time_stop[3]]
    # todo:
    # tmp=train_df.set_index(['DATE','TICKER']).corr().sort_values(['平均卖出收益_1100_1400'],
    #                                             ascending=False).head(80).index
    # train_df=train_df[['DATE','TICKER']+list(tmp)]
    # valid_df = valid_df[['DATE', 'TICKER'] + list(tmp)]
    # test_df=test_df[['DATE','TICKER']+list(tmp)]
    # -------------------------------------------------------------
    train_x,train_y=split_df(train_df,y_col='平均卖出收益_1100_1400')
    valid_x,vaild_y=split_df(valid_df,y_col='平均卖出收益_1100_1400')
    test_x,test_y=split_df(test_df,y_col='平均卖出收益_1100_1400')
    return train_x,train_y,valid_x,vaild_y,test_x,test_y


def ml_para(flag='label1'):
    # model=lgb.LGBMRegressor(subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=12,verbose=-1)
    if flag=='label1':
        # model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.01, max_depth=4, num_leaves=15, min_child_samples=5,
        #                           subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=12, verbose=-1)
        model= XGBRegressor(n_estimators=10000, max_depth=4, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=12)
    # elif flag=='label2':
    #     model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
    #                          random_state=42, n_jobs=12)
    # else:
    #     model = XGBRegressor(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
    #                          random_state=42, n_jobs=12)
    # model = XGBRegressor(n_estimators=5000, max_depth=10, learning_rate=0.01, subsample=0.8, colsample_bytree=0.8,
    #                      random_state=42, n_jobs=12)
    # model=NGBoost(Dist=Normal,n_estimators=1000,learning_rate=0.01)
    # model = lgb.LGBMRegressor(n_estimators=5000, learning_rate=0.01, max_depth=4, num_leaves=15, min_child_samples=5,
    #                           subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=12, verbose=-1)
    # model=MLP_my(97,32,16,1)
    return model

# x

def ml(file_name,time_stop=['20240630','20240701','20241231','20250101'],flag='label1'):
    path = fr'\\192.168.1.210\Factor_Storage\田逸心\calc6临时用\tyx_ml\{file_name}_第一组'
    train_x, train_y, valid_x, valid_y, test_x, test_y = get_data(file_name, time_stop)
    grid_search = ml_para(flag)
    # #  dl参数设置--------------------------------
    # optimizer=torch.optim.SGD(grid_search.parameters(), lr = 0.01)
    # loss_func=nn.MSELoss()
    # train_loss=[]
    # --------------------------------
    grid_search.fit(train_x, train_y)
    # print(grid_search.best_params_)
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
    # os.makedirs(path, exist_ok=True)
    group_mean(test,file_name)
    # print('test ic:', pearsonr(test_pred_y.flatten(), test_y.values.flatten())[0])
    # feather.write_dataframe(train,os.path.join(path,'train.feather'))
    # feather.write_dataframe(valid,os.path.join(path,'valid.feather'))
    # feather.write_dataframe(test,os.path.join(path,'test.feather'))
    # joblib.dump(grid_search, os.path.join(path,'xgb.joblib'))
    # group_mean(test, file_name)
    # print('保存成功')

def get_qcut(x,usename):
    try:
        x['bins'] = pd.qcut(x[usename], 10, labels=False) + 1
        return x
    except:
        print(x['DATE'].values[0])
        return pd.DataFrame()


def group_mean(aa,file_name=''):
    ret=feather.read_dataframe(r'C:\Users\admin\Desktop\ret.feather')
    # ret.loc[ret['平均卖出收益_1100_1400'] > 0.4, '平均卖出收益_1100_1400'] = 0.4
    ret['平均卖出收益_1100_1400'] = ret['平均卖出收益_1100_1400'].clip(-0.4, 0.4)  # 限制在[-0.4, 0.4]
    ret['平均卖出收益_1100_1400'] -= 0.0007  # 统一减去一个基准值
    ret['DATE'] = ret['DATE'].astype(str)
    ret.rename(columns={'平均卖出收益_1100_1400':'ret'},inplace=True)
    # usename = list(np.setdiff1d(aa.columns,['TICKER','DATE']))[0]
    zz = pd.merge(ret, aa, how='inner', on=['TICKER', 'DATE'])
    usename = list(np.setdiff1d(aa.columns, ['TICKER', 'DATE']))[0]

    # 每天切分一个十组，每天买第十组
    zz = zz.groupby('DATE').apply(get_qcut, usename).reset_index(drop=True)

    res = zz.groupby(['DATE', 'bins'])['ret'].mean().reset_index(drop=False)
    res = res.groupby('bins')['ret'].mean().reset_index(drop=False)
    ics = zz.groupby('DATE').apply(lambda x: x[[usename, 'ret']].corr(method='spearman').iloc[0, 1])
    print(res)
    print('IC:', ics.mean(), 'ICIR:', abs(ics.mean() / ics.std()))
    print('*' * 100)

    # res=zz.groupby(['DATE', 'group'])['ret'].mean().reset_index(drop=False).groupby('group')['ret'].mean()
    # res.to_csv(fr'\\192.168.1.210\Factor_Storage\田逸心\calc6临时用\tyx_ml\{file_name}_第一组\test_res.csv',index=False)
    return res



if __name__=='__main__':
    ml('20251115_sz_auction_processed')
    # ml('20251115_sz_auction_processed',['20220630','20220701','20221231','20230101'],'label2')
    # ml('20251115_sz_auction_processed',['20221231','20230101','20230630','20230701'],'label3')
    # print('-------------------')
    # ll=[['20220630','20220701','20221231','20230101'],['20221231','20230101','20230630','20230701']]
    # n=2
    # for l in ll:
    #     ml('20251115_sz_auction_processed',l,f'label{n}')
    #     n+=1
    #     print('----------------------')


