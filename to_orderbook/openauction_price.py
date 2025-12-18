# -*- coding: utf-8 -*-
# Author: PYShuo
# Date: 2025-09-18
# Description: 模拟个股开盘集合竞价期间的撮合价格变动
    
# -*- coding: utf-8 -*-
import pdb 
import numpy as np
import pandas as pd
from collections import defaultdict

class OpenAuctionIOPEngine:
    """
    开盘集合竞价 IOP 模拟器（仅需 df_order / df_cancel + 前收价）
    必要列：timestamp, Price, OrderQty, Side, ApplSeqNum（可用 colmap 重映射）
    价格选择规则：最大成交量 → 最小不平衡 → 最接近前收盘价
    """

    def __init__(self, df_order: pd.DataFrame, df_cancel: pd.DataFrame, prev_close: float, limit_price: set,
                 colmap: dict | None = None):
        self.prev_close = float(prev_close)
        self.low_limit = limit_price[0]
        self.up_limit = limit_price[1]
        self.colmap = {
            "timestamp": "timestamp",
            "price": "Price",
            "qty": "OrderQty",
            "side": "Side",
            "seq": "ApplSeqNum",
        }
        if colmap:
            self.colmap.update(colmap)
        self.orders = self._normalize(df_order, is_cancel=False)
        self.cancels = self._normalize(df_cancel, is_cancel=True)
        self.events = None
        self.events_with_iop = None

    def run(self) -> pd.DataFrame:
        self.events = self._build_events(self.orders, self.cancels)
        self.events_with_iop = self._compute_iop_over_events(self.events, self.prev_close)
        return self.events_with_iop

    # -------------------- internal --------------------

    def _normalize(self, df: pd.DataFrame, is_cancel: bool) -> pd.DataFrame:
        c = self.colmap
        need = [c["timestamp"], c["price"], c["qty"], c["side"], c["seq"]]
        if 'TradeQty' in df.columns:
            df['Side'] = (df['BidApplSeqNum'] > df['OfferApplSeqNum']).astype(str)
            df['ApplSeqNum'] = df['BidApplSeqNum'] + df['OfferApplSeqNum']
            df['OrderQty'] = df['TradeQty']
            df = df.merge(self.orders[['ApplSeqNum', 'Price']], on='ApplSeqNum', how='left', suffixes=['', '_order'])
            df['Price'] = df['Price_order']
        out = df[need].copy()
        if not np.issubdtype(out[c["timestamp"]].dtype, np.datetime64):
            out[c["timestamp"]] = pd.to_datetime(out[c["timestamp"]])

        out = out.rename(columns={
            c["timestamp"]: "timestamp",
            c["price"]: "Price",
            c["qty"]: "qty",
            c["side"]: "Side",
            c["seq"]: "ApplSeqNum",
        })
        out["is_cancel"] = is_cancel
        out["is_buy"] = out["Side"].map(self._side_to_bool).astype(bool)
        out["Price"] = out["Price"].astype(float)
        out["qty"] = out["qty"].astype(float)
        out = out.loc[np.isfinite(out["Price"]) & (out["qty"] > 0)].copy()
        return out[["timestamp", "ApplSeqNum", "Price", "qty", "is_buy", "is_cancel"]]

    @staticmethod
    def _side_to_bool(x) -> bool:
        if pd.isna(x): return False
        s = str(x).strip().lower()
        if s in {"b", "buy", "bid", "1", "true", "True"}: return True
        if s in {"s", "sell", "ask", "-1", "false", 'False'}: return False
        return False

    @staticmethod
    def _build_events(orders: pd.DataFrame, cancels: pd.DataFrame) -> pd.DataFrame:
        ev = pd.concat([orders, cancels], ignore_index=True)
        return ev.sort_values(["timestamp", "ApplSeqNum"]).reset_index(drop=True)

    @staticmethod
    def _choose_by_reference(prices: np.ndarray, ref: float) -> float:
        if prices.size == 1 or not np.isfinite(ref):
            return float(np.median(prices))
        idx = int(np.argmin(np.abs(prices - ref)))
        # 若与参考价等距的并列多个价，再用中位数二次打破
        ties = np.isclose(np.abs(prices - ref), np.abs(prices[idx] - ref))
        if ties.sum() > 1:
            mid = float(np.median(prices))
            idx2 = int(np.argmin(np.abs(prices[ties] - mid)))
            return float(np.sort(prices[ties])[idx2])
        return float(prices[idx])

    def _compute_iop_over_events(self, events: pd.DataFrame, prev_close: float) -> pd.DataFrame:
        buy_book, sell_book = defaultdict(float), defaultdict(float)
        iops, vols, imbs, unmatchs = [], [], [], []
        def _snap(x):      # 对齐到最小价位
            return round(round(x / 0.01) * 0.01, 2)
        def _next_up(x):   # 高于x的一档（含涨停裁剪）
            y = _snap(x + 0.01)
            return min(y, self.up_limit)
        def _prev_down(x): # 低于x的一档（含跌停裁剪）
            y = _snap(x - 0.01)
            return max(y, self.low_limit)
        def current_iop():
            if not buy_book and not sell_book:
                return np.nan, 0.0, 0.0, 0.0
            prices_buy = np.fromiter(buy_book.keys(), dtype=float) if buy_book else np.array([], float)
            prices_sell = np.fromiter(sell_book.keys(), dtype=float) if sell_book else np.array([], float)
            if prices_buy.size == 0 and prices_sell.size > 0:
                smin = float(np.min(prices_sell))
                cand = _snap(self.prev_close)
                if cand >= smin:
                    cand = _prev_down(smin)           # 只能取低于最低卖的一侧
                return float(cand), 0.0, -float(sum(sell_book.values())), float(sum(sell_book.values()))

            elif prices_sell.size == 0 and prices_buy.size > 0:
                bmax = float(np.max(prices_buy))
                cand = _snap(self.prev_close)
                if cand <= bmax:
                    cand = _next_up(bmax)             # 只能取高于最高买的一侧
                return float(cand), 0.0, float(sum(buy_book.values())), float(sum(buy_book.values()))

            elif prices_buy.size > 0 and prices_sell.size > 0 and float(np.max(prices_buy)) < float(np.min(prices_sell)):
                bmax, smin = float(np.max(prices_buy)), float(np.min(prices_sell))
                c_hi = _snap(self.prev_close)
                if c_hi <= bmax:
                    c_hi = _next_up(bmax)             # > bmax 的最近一档
                c_lo = _snap(self.prev_close)
                if c_lo >= smin:
                    c_lo = _prev_down(smin)           # < smin 的最近一档
                chosen = c_hi if abs(c_hi - self.prev_close) <= abs(c_lo - self.prev_close) else c_lo
                return float(chosen), 0.0, float(sum(buy_book.values()) - sum(sell_book.values())), float(sum(buy_book.values()) + sum(sell_book.values()))

            candidates = np.unique(np.concatenate([prices_buy, prices_sell]))

            buy_df = (pd.DataFrame({"price": list(buy_book.keys()),
                                    "qty": list(buy_book.values())})
                      .sort_values("price", ascending=False))
            buy_df["cum"] = buy_df["qty"].cumsum()

            sell_df = (pd.DataFrame({"price": list(sell_book.keys()),
                                     "qty": list(sell_book.values())})
                       .sort_values("price", ascending=True))
            sell_df["cum"] = sell_df["qty"].cumsum()

            buy_map = pd.merge_asof(pd.DataFrame({"price": candidates}).sort_values("price"),
                                    buy_df[["price", "cum"]].sort_values("price"),
                                    on="price", direction="forward")\
                        .set_index("price")["cum"].fillna(0.0)
            sell_map = pd.merge_asof(pd.DataFrame({"price": candidates}).sort_values("price"),
                                     sell_df[["price", "cum"]].sort_values("price"),
                                     on="price", direction="backward")\
                         .set_index("price")["cum"].fillna(0.0)
            traded = np.minimum(buy_map.values, sell_map.values)
            if traded.max() <= 0:
                return np.nan, 0.0, float(buy_map.max() - sell_map.max()), float(buy_map.max() + sell_map.max())

            max_vol = traded.max()
            mask = (traded == max_vol)
            p_cand = candidates[mask]
            imb = (buy_map.values - sell_map.values)[mask]

            abs_imb = np.abs(imb)
            keep = (abs_imb == abs_imb.min())
            p_cand, imb = p_cand[keep], imb[keep]

            if p_cand.size > 1:
                chosen = self._choose_by_reference(p_cand, prev_close)
                idx = int(np.where(p_cand == chosen)[0][0])
                return float(chosen), float(max_vol), float(imb[idx]), float(abs(imb[idx]))
            return float(p_cand[0]), float(max_vol), float(imb[0]), float(abs(imb[0]))
        
        for _, r in events.iterrows():
            px = float(r["Price"]); q = float(r["qty"]); buy = bool(r["is_buy"])
            is_cancel = bool(r["is_cancel"])

            if buy:
                buy_book[px] += (-q if is_cancel else q)
                if buy_book[px] <= 0: buy_book.pop(px, None)
            else:
                sell_book[px] += (-q if is_cancel else q)
                if sell_book[px] <= 0: sell_book.pop(px, None)
            iop, vol, imb, unmatch_vol = current_iop()
            iops.append(iop); vols.append(vol); imbs.append(imb); unmatchs.append(unmatch_vol)

        out = events.copy()
        out["iop"] = iops
        out["iop_match_qty"] = vols
        out["iop_unmatch_qty"] = unmatchs
        out["iop_imbalance"] = imbs
        return out


# 用法示例（仅示意）：
# engine = OpenAuctionIOPEngine(df_order, df_cancel, prev_close=pre_close,
#                               colmap={"timestamp":"TransactTime","price":"Price","qty":"OrderQty",
#                                       "side":"Side","seq":"ApplSeqNum"})
# events_with_iop = engine.run()



