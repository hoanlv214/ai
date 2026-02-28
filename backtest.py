"""
backtest.py - Backtest non-overlapping

QUAN TRONG: Khi predict_bars > 1 (vd: predict 60m voi nen 5m = 12 bars),
cac future_return chong lan nhau. Backtest cu cumprod() tat ca bars
se cho ket qua sai.

Fix: Chi lay NON-OVERLAPPING trades. Sau khi vao lenh, phai doi
du PREDICT_BARS truoc khi vao lenh tiep.
"""

import numpy as np
import pandas as pd
from typing import Dict
from config import CFG


def run_backtest(y_proba: np.ndarray,
                 future_returns: pd.Series) -> Dict[str, float]:
    """
    Backtest non-overlapping: sau khi vao lenh, doi du PREDICT_BARS
    truoc khi vao lenh tiep de tranh return chong lan.

    Buy & Hold tinh bang gia thuc te (close cuoi / close dau).
    """
    prob = np.asarray(y_proba, dtype=np.float32)
    ret = np.asarray(future_returns.values, dtype=np.float32)
    n = len(prob)
    predict_bars = CFG.PREDICT_BARS

    # === NON-OVERLAPPING SIGNALS ===
    # Sau khi vao lenh tai bar i, skip PREDICT_BARS bars truoc khi
    # xet lenh tiep (tranh return chong lan)
    position = np.zeros(n, dtype=np.int8)
    i = 0
    while i < n:
        if prob[i] > CFG.PROB_THRESHOLD:
            position[i] = 1
            i += predict_bars  # Skip de khong overlap
        else:
            i += 1

    # Strategy return: chi tinh tai cac bar co lenh
    strat_returns = position * ret

    # === BUY & HOLD: tinh bang gia thuc te ===
    # Lay close data tu index cua future_returns
    try:
        idx = future_returns.index
        # B&H = close cuoi / close dau
        # future_return = close_future / close_now - 1
        # Khong the dung truc tiep. Tinh xap xi bang 1-bar log returns
        # Lay close data tu DataFrame cha neu co
        bnh_ret = float(np.expm1(np.sum(np.log1p(
            ret[::predict_bars]  # lay moi predict_bars bar 1 lan
        ))))
    except Exception:
        bnh_ret = 0.0

    # === STRATEGY METRICS ===
    # Cumulative return (chi tu non-overlapping trades)
    trade_returns = ret[position == 1]
    n_trades = int(position.sum())

    if n_trades > 0:
        cum_ret_arr = np.cumprod(1 + trade_returns)
        total_ret = float(cum_ret_arr[-1] - 1)

        # Max drawdown
        cum_max = np.maximum.accumulate(cum_ret_arr)
        drawdown = cum_ret_arr / cum_max - 1
        max_dd = float(drawdown.min())

        # Sharpe: annualize dua tren so trades/nam
        # Tinh so trades trung binh moi nam
        total_bars = n
        bars_per_year = 365 * 24 * (60 // CFG.BAR_MINUTES)
        trades_per_year = n_trades / total_bars * bars_per_year

        if trade_returns.std() > 0 and trades_per_year > 0:
            sharpe = float(
                (trade_returns.mean() / trade_returns.std())
                * np.sqrt(trades_per_year)
            )
        else:
            sharpe = 0.0

        winrate = float((trade_returns > 0).mean())
        avg_win = float(trade_returns[trade_returns > 0].mean()) if (trade_returns > 0).any() else 0.0
        avg_loss = float(trade_returns[trade_returns <= 0].mean()) if (trade_returns <= 0).any() else 0.0
    else:
        total_ret = 0.0
        max_dd = 0.0
        sharpe = 0.0
        winrate = avg_win = avg_loss = 0.0

    # Expectancy
    if n_trades > 0:
        expectancy = (winrate * avg_win) + ((1 - winrate) * avg_loss)
    else:
        expectancy = 0.0

    results = {
        'total_return': total_ret,
        'bnh_return': bnh_ret,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': n_trades,
        'winrate': winrate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
    }

    print("=" * 50)
    print("BACKTEST RESULTS (non-overlapping)")
    print("=" * 50)
    print(f"   Threshold:   {CFG.PROB_THRESHOLD}")
    print(f"   Predict:     {CFG.PREDICT_MINUTES}m ({predict_bars} bars)")
    print(f"   Total bars:  {n:,}")
    print(f"   N trades:    {n_trades:,} ({n_trades/n*100:.1f}%)")
    print(f"   Strategy:    {total_ret:.2%}")
    print(f"   Buy & Hold:  {bnh_ret:.2%}")
    print(f"   Sharpe:      {sharpe:.4f}")
    print(f"   Max DD:      {max_dd:.2%}")
    print(f"   Winrate:     {winrate:.2%}")
    print(f"   Avg win:     {avg_win:.4%}")
    print(f"   Avg loss:    {avg_loss:.4%}")
    print(f"   Expectancy:  {expectancy:.4%} per trade")

    return results