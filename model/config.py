"""exp04 model-side configuration: the experiment grid and the expanding-window split.

The paper's protocol: quarterly EXPANDING window. Test quarters Q2-2022 … Q2-2023 (5 quarters);
for each test quarter, validation = the immediately preceding quarter, training = everything before
that. Sampling/labeling params stay FIXED across quarters (no per-quarter re-tuning).
"""
from __future__ import annotations

from dataclasses import dataclass, field

WINDOW = 96                       # input sequence length (bars)
VERTICAL = 24                     # triple-barrier vertical timeout (bars forward)
TEST_QUARTERS = ["2022Q2", "2022Q3", "2022Q4", "2023Q1", "2023Q2"]   # paper Fig. 4

# 60/40 trade filter + costs (paper)
LONG_TH, SHORT_TH = 0.60, 0.40
COST = 0.001                      # 0.1% per side
RISK_FREE = 0.0314                # test-period T-bill avg, for annualized Sharpe


def quarter_ord(t_open_ms) -> int:
    """Monotone quarter ordinal from epoch-ms (year*4 + quarter_index)."""
    import numpy as np
    import pandas as pd
    ts = pd.to_datetime(np.asarray(t_open_ms), unit="ms", utc=True)
    return (ts.year * 4 + (ts.quarter - 1)).to_numpy()


def parse_quarter(q: str) -> int:
    y, qq = q.split("Q")
    return int(y) * 4 + (int(qq) - 1)


@dataclass
class ExperimentConfig:
    coin: str = "ETHUSDT"
    bartype: str = "cusum"                 # cusum | range
    threshold: float = 0.02
    label_scheme: str = "triple_barrier"   # triple_barrier | next_bar
    barrier: float = 0.05                  # triple-barrier width (paper: 5% for ETH CUSUM)
    vertical: int = VERTICAL
    model: str = "resnet_lstm"
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


# v1 scoped reproduction (CPU-feasible shape; runs on the GPU box):
# {ETH,BTC} x {cusum,range} x {2%} x triple_barrier x resnet_lstm x 3 seeds x 5 quarters
def v1_grid() -> list[ExperimentConfig]:
    out = []
    for coin in ("ETHUSDT", "BTCUSDT"):
        for bartype in ("cusum", "range"):
            out.append(ExperimentConfig(coin=coin, bartype=bartype, threshold=0.02,
                                        barrier=0.05 if coin == "ETHUSDT" else 0.025))
    return out
