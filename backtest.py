"""
backtest.py - Backtest don gian
Params doc tu config.py.
"""

import numpy as np
import pandas as pd
from typing import Dict
from config import CFG


def run_backtest(y_proba: np.ndarray,
                 future_returns: pd.Series) -> Dict[str, float]:
    """
    Backtest: long khi prob > CFG.PROB_THRESHOLD.
    Tat ca params tu config.py.
    """
    prob = np.asarray(y_proba, dtype=np.float32)
    ret = np.asarray(future_returns.values, dtype=np.float32)

    position = (prob > CFG.PROB_THRESHOLD).astype(np.int8)
    strat_ret = position * ret

    # Cumulative return
    cum_ret = np.cumprod(1 + strat_ret)
    cum_bnh = np.cumprod(1 + ret)

    # Sharpe ratio (annualized)
    bars_per_year = 365 * 24 * (60 // CFG.BAR_MINUTES)
    if strat_ret.std() > 0:
        sharpe = float((strat_ret.mean() / strat_ret.std()) * np.sqrt(bars_per_year))
    else:
        sharpe = 0.0

    # Max drawdown
    cum_max = np.maximum.accumulate(cum_ret)
    drawdown = cum_ret / cum_max - 1
    max_dd = float(drawdown.min())

    total_ret = float(cum_ret[-1] - 1)
    bnh_ret = float(cum_bnh[-1] - 1)

    n_trades = int(position.sum())
    trade_mask = position == 1
    if n_trades > 0:
        trade_returns = ret[trade_mask]
        winrate = float((trade_returns > 0).mean())
        avg_win = float(trade_returns[trade_returns > 0].mean()) if (trade_returns > 0).any() else 0.0
        avg_loss = float(trade_returns[trade_returns <= 0].mean()) if (trade_returns <= 0).any() else 0.0
    else:
        winrate = avg_win = avg_loss = 0.0

    results = {
        'total_return': total_ret,
        'bnh_return': bnh_ret,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'n_trades': n_trades,
        'winrate': winrate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }

    print("=" * 50)
    print("BACKTEST RESULTS")
    print("=" * 50)
    print(f"   Threshold:   {CFG.PROB_THRESHOLD}")
    print(f"   Total bars:  {len(prob):,}")
    print(f"   N trades:    {n_trades:,} ({n_trades/len(prob)*100:.1f}%)")
    print(f"   Strategy:    {total_ret:.2%}")
    print(f"   Buy & Hold:  {bnh_ret:.2%}")
    print(f"   Sharpe:      {sharpe:.4f}")
    print(f"   Max DD:      {max_dd:.2%}")
    print(f"   Winrate:     {winrate:.2%}")
    print(f"   Avg win:     {avg_win:.4%}")
    print(f"   Avg loss:    {avg_loss:.4%}")

    return results