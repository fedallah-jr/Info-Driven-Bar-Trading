"""Model-side configuration: experiment grid + the expanding-window split.

Protocol: quarterly EXPANDING window. For each test quarter, validation = the immediately preceding
quarter, training = everything before that. Sampling/labeling params stay FIXED across quarters.
"""
from __future__ import annotations

from dataclasses import dataclass

WINDOW = 96                       # input sequence length (bars)
VERTICAL = 24                     # triple-barrier vertical timeout (bars forward)
TEST_QUARTERS = ["2022Q2", "2022Q3", "2022Q4", "2023Q1", "2023Q2"]   # the paper's window

# 60/40 trade filter + costs (paper)
LONG_TH, SHORT_TH = 0.60, 0.40
COST = 0.001                      # 0.1% per side
RISK_FREE = 0.0314                # test-period T-bill avg, for annualized Sharpe

BASKET = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
          "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "LTCUSDT"]


def quarter_ord(t_open_ms) -> int:
    """Monotone quarter ordinal from epoch-ms (year*4 + quarter_index)."""
    import numpy as np
    import pandas as pd
    ts = pd.to_datetime(np.asarray(t_open_ms), unit="ms", utc=True)
    return (ts.year * 4 + (ts.quarter - 1)).to_numpy()


def parse_quarter(q: str) -> int:
    y, qq = q.split("Q")
    return int(y) * 4 + (int(qq) - 1)


def quarters(start: str = "2022Q2", end: str = "2026Q1") -> list[str]:
    """Inclusive list of quarter labels start..end."""
    s, e = parse_quarter(start), parse_quarter(end)
    return [f"{o // 4}Q{o % 4 + 1}" for o in range(s, e + 1)]


# Extended OOS window: the paper's 5 quarters PLUS everything through 2026-Q1 — data we have but
# never trained on or tuned against. This is the real held-out robustness test (~3x more trades).
EXTENDED_QUARTERS = quarters("2022Q2", "2026Q1")


@dataclass
class ExperimentConfig:
    coin: str = "ETHUSDT"
    bartype: str = "cusum"                 # cusum | range
    threshold: float = 0.02
    label_scheme: str = "triple_barrier"   # triple_barrier | next_bar
    barrier: float = 0.05                  # triple-barrier width
    vertical: int = VERTICAL
    model: str = "resnet_lstm"             # resnet_lstm | tsmixer | transformer
    window: int = WINDOW
    seeds: tuple[int, ...] = (0, 1, 2)
    test_quarters: tuple[str, ...] = tuple(TEST_QUARTERS)
    epochs: int = 100
    batch_size: int = 256
    patience: int = 10
    lr: float = 1e-3

    def dataset_path(self, root: str = "data/datasets") -> str:
        return f"{root}/{self.coin}_{self.bartype}_thr{self.threshold:.3f}.parquet"

    def tag(self) -> str:
        return f"{self.coin}_{self.bartype}_thr{self.threshold:.3f}_{self.label_scheme}_{self.model}"


def v1_grid() -> list[ExperimentConfig]:
    """Paper reproduction: {ETH,BTC} x {cusum,range} x 2% x triple_barrier x resnet_lstm, paper window."""
    out = []
    for coin in ("ETHUSDT", "BTCUSDT"):
        for bartype in ("cusum", "range"):
            out.append(ExperimentConfig(coin=coin, bartype=bartype, threshold=0.02,
                                        barrier=0.05 if coin == "ETHUSDT" else 0.025))
    return out


def basket_grid(bartype="cusum", threshold=0.02, barrier=0.04, model="resnet_lstm",
                seeds=(0, 1, 2), test_quarters=None) -> list[ExperimentConfig]:
    """Per-coin configs over the full 10-coin basket, extended OOS by default. Single barrier — the
    paper notes barriers don't transfer across coins, so treat non-ETH/BTC results as exploratory."""
    q = tuple(test_quarters or EXTENDED_QUARTERS)
    return [ExperimentConfig(coin=c, bartype=bartype, threshold=threshold, barrier=barrier,
                             model=model, seeds=tuple(seeds), test_quarters=q) for c in BASKET]
