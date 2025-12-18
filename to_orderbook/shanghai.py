import numpy as np
import pandas as pd
from tqdm import tqdm


def gen_orderbook():
    file_df = {}
    df=pd.read_feather(r'C:\Users\Administrator\Desktop\临时用\临时用\stock_tick\600000.feather')
    df=df[df.Type!='S']
    df.sort_values('TickIndex',inplace=True)
    # df['OrigTime']=df.TickTime.astype(str).str[:14] #.astype(int)
    df['OrigTime']=df.TickTime #.astype(int)
    order_book = {'bid': {'bid_price': [], 'bid_qty': [], 'bid_num': []},
                  'offer': {'offer_price': [], 'offer_qty': [], 'offer_num': []}}
    for t in sorted(set(df['OrigTime'])):
        tmp_df=df[df.OrigTime==t]
        for _,r in tmp_df.iterrows():
            bookno = max(r.BuyOrderNo, r.SellOrderNo)
            label = 'bid' if r.BuyOrderNo > r.SellOrderNo else 'offer'
            if r.Type=='A': # 委托数据
                # 限价单处理逻辑
                if r.Price not in order_book[label][f'{label}_price']:
                    # 当前订单簿无此价格，append进去
                    order_book[label][f'{label}_price'].append(r.Price)
                    order_book[label][f'{label}_qty'].append([r.Qty])
                    order_book[label][f'{label}_num'].append([bookno])
                else:
                    # 当前订单簿有此价格，把单号和qty加进去
                    idx=order_book[label][f'{label}_price'].index(r.Price)
                    order_book[label][f'{label}_qty'][idx].append(r.Qty)
                    order_book[label][f'{label}_num'][idx].append(bookno)
             #撤单必有一个单号是0，成交也是单号比大小
            elif r.Type=='D': # 撤销
                delidx = [x for x in order_book[label][f'{label}_num'] if bookno in x][0]
                # 选出待要删除的price对应的applseqnum序列了
                idx = order_book[label][f'{label}_num'].index(delidx)  # 返回该applseqnum序列的index
                no_idx = order_book[label][f'{label}_num'][idx].index(bookno)
                del order_book[label][f'{label}_num'][idx][no_idx]
                del order_book[label][f'{label}_qty'][idx][no_idx]
                if order_book[label][f'{label}_qty'][idx] == []:
                    del order_book[label][f'{label}_price'][idx]
                    del order_book[label][f'{label}_qty'][idx]
                    del order_book[label][f'{label}_num'][idx]

            elif r.Type=='T': # 成交
                if r.TickBSFlag=='N':
                    bidno, offerno=r.BuyOrderNo ,r.SellOrderNo
                    for label, labelno in zip(['bid', 'offer'], [bidno, offerno]):
                        label_ = [x for x in order_book[label][f'{label}_num'] if labelno in x][0]# 找到目标序列
                        label_index = order_book[label][f'{label}_num'].index(label_)  # 找到目标序列的index
                        label_num_idx = order_book[label][f'{label}_num'][label_index].index(labelno)  # 找目标在目标序列中的idx
                        order_book[label][f'{label}_qty'][label_index][label_num_idx] -= min(
                            order_book[label][f'{label}_qty'][label_index][label_num_idx], r.Qty)
                        # 减去 min(目标index的对应的qty,这笔要成交的qty)
                        if np.isclose(order_book[label][f'{label}_qty'][label_index][label_num_idx],0):
                            # 如果qty接近0，那么要删掉这个qty
                            del order_book[label][f'{label}_qty'][label_index][label_num_idx]
                            del order_book[label][f'{label}_num'][label_index][label_num_idx]
                        if len(order_book[label][f'{label}_num'][label_index])==0:
                            del order_book[label][f'{label}_num'][label_index]
                            del order_book[label][f'{label}_qty'][label_index]
                            del order_book[label][f'{label}_price'][label_index]
                else: #B/S
                    oppo_label='bid' if label=='offer' else 'offer'
                    labelno=min(r.BuyOrderNo,r.SellOrderNo)
                    label_=[x for x in order_book[oppo_label][f'{oppo_label}_num'] if labelno in x][0]
                    label_index=order_book[oppo_label][f'{oppo_label}_num'].index(label_)
                    label_num_idx=order_book[oppo_label][f'{oppo_label}_num'][label_index].index(labelno)
                    order_book[oppo_label][f'{oppo_label}_qty'][label_index][label_num_idx] -= min(
                        order_book[oppo_label][f'{oppo_label}_qty'][label_index][label_num_idx],r.Qty)
                    # 减去 min(目标index的对应的qty,这笔要成交的qty)
                    if np.isclose(order_book[oppo_label][f'{oppo_label}_qty'][label_index][label_num_idx],0):
                        del order_book[oppo_label][f'{oppo_label}_qty'][label_index][label_num_idx]
                        del order_book[oppo_label][f'{oppo_label}_num'][label_index][label_num_idx]
                    if len(order_book[oppo_label][f'{oppo_label}_num'][label_index]) == 0:
                        del order_book[oppo_label][f'{oppo_label}_num'][label_index]
                        del order_book[oppo_label][f'{oppo_label}_qty'][label_index]
                        del order_book[oppo_label][f'{oppo_label}_price'][label_index]
            # if t == 20250627093000990:
        order_book=sort_dict(order_book,'bid')
        order_book=sort_dict(order_book,'offer')
        # print('OrigTime=', t)
        # try:
        #     print('bid:', order_book['bid']['bid_price'][0], ' ', sum(order_book['bid']['bid_qty'][0]))
        #     print('offer:', order_book['offer']['offer_price'][0], ' ', sum(order_book['offer']['offer_qty'][0]))
        # except:
        #     pass
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
                new_row[f'BidNum{i}'] = len(order_book['bid']['bid_num'][i - 1])
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
    file_df = pd.DataFrame(file_df).T.reset_index(drop=True)
    file_df.to_feather(r'C:\Users\Administrator\Desktop\order_book_600000.feather')

def sort_dict(order_book,label):
    if len(order_book[label][f'{label}_price'])>1:
        # 对订单簿按照价格排序
        sort_index=sorted(range(len(order_book[label][f'{label}_price'])), key=lambda i: order_book[label][f'{label}_price'][i], reverse=label=='bid')
        # 买盘以高价优先,为了能够快速交易
        for k in list(order_book[label].keys()):
            order_book[label][k]=[order_book[label][k][i] for i in sort_index]
    return order_book

if __name__=='__main__':
    gen_orderbook()