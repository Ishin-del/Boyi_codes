import numpy as np
import pandas as pd
from tqdm import tqdm


def gen_orderbook(code):
    df=read_data(code)
    order_book = {'bid': {'bid_price': [], 'bid_qty': [], 'bid_num': []},
                  'offer': {'offer_price': [], 'offer_qty': [], 'offer_num': []}}
    # base_columns = ['OrigTime', 'SecurityID']
    # price_columns = [f'{side}PX{i}' for i in range(1, 21) for side in ['Offer', 'Bid']]
    # size_columns = [f'{side}Size{i}' for i in range(1, 21) for side in ['Offer', 'Bid']]
    # num_columns = [f'{side}Num{i}' for i in range(1, 21) for side in ['Offer', 'Bid']]
    # columns = base_columns + [col for pair in zip(price_columns, size_columns,num_columns) for col in pair]
    # file_df=pd.DataFrame(columns=columns)
    file_df = {}
    wrong_order_list=[]
    for t in tqdm(sorted(set(df['OrigTime']))):
        tmp_df=df[df.OrigTime==t]
        for _,r in tmp_df.iterrows():
            book_class = 'order' if str(r.ExecType) == 'nan' else 'trade'
            if book_class=='order': # 委托数据
                r.Side=int(r.Side)
                r.OrderType = str(r.OrderType)
                label = 'bid' if r.Side == 1 else 'offer'  # 判断买卖方向
                if r.OrderType=='2':
                    # 限价单处理逻辑
                    if r.Price not in order_book[label][f'{label}_price']:
                        # 当前订单簿无此价格，append进去
                        order_book[label][f'{label}_price'].append(r.Price)
                        order_book[label][f'{label}_qty'].append([r.OrderQty])
                        order_book[label][f'{label}_num'].append([r.ApplSeqNum])
                    else:
                        # 当前订单簿有此价格，把单号和qty加进去
                        idx=order_book[label][f'{label}_price'].index(r.Price)
                        order_book[label][f'{label}_qty'][idx].append(r.OrderQty)
                        order_book[label][f'{label}_num'][idx].append(r.ApplSeqNum)
                elif r.OrderType=='1': #市价单
                    oppo_label='bid' if label=='offer' else 'offer'
                    """以对手方卖一买一的价格作为price加进本方"""
                    if order_book[oppo_label][f'{oppo_label}_price']!=[]:
                        p=max(order_book[oppo_label][f'{oppo_label}_price']) if oppo_label=='bid' else min(order_book[oppo_label][f'{oppo_label}_price'])
                        order_book[label][f'{label}_price'].append(p)
                        order_book[label][f'{label}_qty'].append([r.OrderQty])
                        order_book[label][f'{label}_num'].append([r.ApplSeqNum])
                    else:
                        wrong_order_list.append(r.ApplSeqNum)

                elif r.OrderType=='U': #本方最优
                    if order_book[label][f'{label}_price']!=[]:
                        p=max(order_book[label][f'{label}_price']) if label=='bid' else min(order_book[label][f'{label}_price'])
                        idx=order_book[label][f'{label}_price'].index(p)
                        order_book[label][f'{label}_qty'][idx].append(r.OrderQty)
                        order_book[label][f'{label}_num'][idx].append(r.ApplSeqNum)
                    else:
                        wrong_order_list.append(r.ApplSeqNum)
            elif book_class=='trade': # 成交数据
                if (r.BidApplSeqNum in wrong_order_list) or (r.OfferApplSeqNum in wrong_order_list):
                    continue
                r.ExecType=str(r.ExecType)
                label='bid' if r.BidApplSeqNum>r.OfferApplSeqNum else 'offer' #撤单必有一个单号是0，成交也是单号比大小
                if r.ExecType=='4': # 撤销
                    bookno = r.BidApplSeqNum if label == 'bid' else r.OfferApplSeqNum
                    """
                    先找到待删除applseqnum在哪一对应的seq_num序列里面，找到该序列的index,再找到该applseqnum在序列里的index,
                    用这两个index确定位置，进行删除;
                    """
                    delidx=[x for x in order_book[label][f'{label}_num'] if bookno in x][0]
                    #选出待要删除的price对应的applseqnum序列了
                    idx=order_book[label][f'{label}_num'].index(delidx) # 返回该applseqnum序列的index
                    no_idx=order_book[label][f'{label}_num'][idx].index(bookno)
                    del order_book[label][f'{label}_num'][idx][no_idx]
                    del order_book[label][f'{label}_qty'][idx][no_idx]
                    if order_book[label][f'{label}_qty'][idx]==[]:
                        del order_book[label][f'{label}_price'][idx]
                        del order_book[label][f'{label}_qty'][idx]
                        del order_book[label][f'{label}_num'][idx]
                elif r.ExecType=='F': # 成交
                    bidno,offerno=r.BidApplSeqNum,r.OfferApplSeqNum
                    qty=r.TradeQty
                    for label, labelno in zip(['bid', 'offer'], [bidno, offerno]):
                        label_ = [x for x in order_book[label][f'{label}_num'] if labelno in x][0]# 找到目标序列
                        label_index = order_book[label][f'{label}_num'].index(label_)  # 找到目标序列的index
                        label_num_idx = order_book[label][f'{label}_num'][label_index].index(labelno)  # 找目标在目标序列中的idx
                        order_book[label][f'{label}_qty'][label_index][label_num_idx] -= min(
                            order_book[label][f'{label}_qty'][label_index][label_num_idx], qty)
                        # 减去 min(目标index的对应的qty,这笔要成交的qty)
                        if np.isclose(order_book[label][f'{label}_qty'][label_index][label_num_idx],0):
                            # 如果qty接近0，那么要删掉这个qty
                            del order_book[label][f'{label}_qty'][label_index][label_num_idx]
                            del order_book[label][f'{label}_num'][label_index][label_num_idx]
                        if len(order_book[label][f'{label}_num'][label_index])==0:
                            del order_book[label][f'{label}_num'][label_index]
                            del order_book[label][f'{label}_qty'][label_index]
                            del order_book[label][f'{label}_price'][label_index]

        # if t == 20250627093000990:
        order_book=sort_dict(order_book,'bid')
        order_book=sort_dict(order_book,'offer')
        # =======================================================
        # 数据生成文件
        new_row = {}
        new_row['OrigTime'] = t
        new_row['SecurityID'] = '000001'
        # 填充买价和买量 (BidPX1-BidPX20, BidSize1-BidSize20)
        for i in range(1, 21):
            if i <= len(order_book['bid']['bid_price']):
                new_row[f'BidPX{i}'] = order_book['bid']['bid_price'][i - 1]
                # 取每个价位的第一档数量
                new_row[f'BidSize{i}'] = sum(order_book['bid']['bid_qty'][i - 1]) if order_book['bid']['bid_qty'][
                    i - 1] else 0
                new_row[f'BidNum{i}']=len(order_book['bid']['bid_num'][i - 1])
            else:
                new_row[f'BidPX{i}'] = 0
                new_row[f'BidSize{i}'] = 0
                new_row[f'BidNum{i}'] = 0
        # 填充卖价和卖量 (OfferPX1-OfferPX20, OfferSize1-OfferSize20)
        for i in range(1, 21):
            if i <= len(order_book['offer']['offer_price']):
                new_row[f'OfferPX{i}'] = order_book['offer']['offer_price'][i - 1]
                # 取每个价位的第一档数量
                new_row[f'OfferSize{i}'] = sum(order_book['offer']['offer_qty'][i - 1]) if \
                order_book['offer']['offer_qty'][i - 1] else 0
                new_row[f'OfferNum{i}'] = len(order_book['offer']['offer_num'][i - 1])
            else:
                new_row[f'OfferPX{i}'] = 0
                new_row[f'OfferSize{i}'] = 0
                new_row[f'OfferNum{i}'] = 0
        file_df[t] = new_row
        # file_df.loc[len(file_df)] = new_row
        # print('OrigTime=', t)
        # print('bid:', order_book['bid']['bid_price'][0], ' ', sum(order_book['bid']['bid_qty'][0]))
        # print('offer:', order_book['offer']['offer_price'][0], ' ', sum(order_book['offer']['offer_qty'][0]))
    file_df = pd.DataFrame(file_df).T.reset_index(drop=True)
    file_df.to_feather(r'C:\Users\Administrator\Desktop\order_book_000001.feather')

def read_data(path_code,p='C:/Users/Administrator/Desktop/临时用/临时用/'):
    tmp_trade=pd.read_feather(p+'hq_trade_spot/'+path_code)
    tmp_order=pd.read_feather(p+'hq_order_spot/'+path_code)
    tmp_order_ = tmp_order[['ApplSeqNum', 'Price', 'OrderQty', 'Side', 'OrderType', 'OrigTime']]
    tmp_trade_ = tmp_trade[['ApplSeqNum', 'Price', 'TradeQty', 'ExecType', 'BidApplSeqNum', 'OfferApplSeqNum', 'OrigTime']]
    temp = pd.concat([tmp_order_, tmp_trade_]).sort_values(by=['ApplSeqNum']).reset_index(drop=True)
    return temp

def sort_dict(order_book,label):
    if len(order_book[label][f'{label}_price'])>1:
        # 对订单簿按照价格排序
        sort_index=sorted(range(len(order_book[label][f'{label}_price'])), key=lambda i: order_book[label][f'{label}_price'][i], reverse=label=='bid')
        # 买盘以高价优先,为了能够快速交易
        for k in list(order_book[label].keys()):
            order_book[label][k]=[order_book[label][k][i] for i in sort_index]
    return order_book

if __name__=='__main__':
    gen_orderbook('000001.feather')