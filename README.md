# DeepLOB: reproduction and extensions

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/d9nk/deeplob-project/blob/main/DeepLOB.ipynb)

Research seminar project, HSE University. We re-implement and test Zhang, Zohren, Roberts, [*DeepLOB: Deep Convolutional Neural Networks for Limit Order Books*](https://arxiv.org/abs/1808.03668), IEEE Transactions on Signal Processing 67(11), 2019: a CNN → Inception → LSTM network that predicts the direction of the mid-price k events ahead from the last 100 states of a 10-level limit order book.

## Team

Kurbatov Daniil, Kuznetsov Georgiy, Gusev Dmitriy, Sorokin Grigoriy

## Project

We wrote DeepLOB in PyTorch from the paper and reproduced its Table II (FI-2010 benchmark, horizons k = 10 / 20 / 50) with 5 seeds. On top of that we ran a block ablation with linear baselines, a new dataset (Binance BTCUSDT order book, transfer to ETHUSDT), a leave-one-stock-out transfer test inside FI-2010, the paper's trading simulation with transaction costs, and checks for leakage and label sensitivity.

FI-2010 test days 8–10, mean ± std over 5 seeds, %:

| k | paper Acc / F1 (weighted) | ours Acc / F1 (weighted) | ours F1 (macro) | largest test class |
|---|---|---|---|---|
| 10 | 84.47 / 83.40 | 83.51 ± 0.23 / 82.45 ± 0.24 | 72.59 ± 0.45 | 70.69 |
| 20 | 74.85 / 72.82 | 74.93 ± 0.16 / 73.01 ± 0.20 | 64.53 ± 0.33 | 62.08 |
| 50 | 80.51 / 80.35 | 78.21 ± 0.59 / 78.14 ± 0.55 | 76.48 ± 0.58 | 47.31 |

- Table II is reproduced within 1–2 pp, but not from the text of the paper alone: with the published optimiser (Adam, ε = 1) and the PyTorch default initialisation the network stays at the class prior in all 15 runs. It trains with the Keras default initialisation and z-scored inputs, which the paper does not mention.
- The paper reports weighted F1 on labels that are 71 % 'stationary' at k = 10; macro-F1 is 10 pp lower. The independent LOBCAST benchmark reports a failed replication of DeepLOB, but it computes macro-F1, and our macro-F1 matches it.
- Ablation: the convolutional stack carries the result (−9 pp without it), the Inception module adds nothing measurable.
- On Binance BTCUSDT the model reaches macro-F1 64.0 (majority 16.0); transfer to ETHUSDT drops to 46.6. Inside FI-2010 transfer to an unseen stock costs 7.9 pp with the dataset's pooled z-score and 1.5 pp with per-stock normalisation.
- The trading profit of the paper (+0.45 bps per trade at the mid-price) turns into −15.8 bps once the spread is crossed.

## Files

| File | Purpose |
|---|---|
| `DeepLOB.ipynb` | All the code with saved outputs: the authors' reference code, our implementation, data checks, experiments, final tables and figures |
| `data/fi2010_decpre.zip` | FI-2010 benchmark (Ntakaris et al., 2018, CC BY 4.0), the NoAuction DecPre files as published by the DeepLOB authors |
| `results/full_runs.zip` | Metrics, per-epoch histories and training logs of the 152 full-length runs behind the reported numbers |
| `requirements.txt` | Packages for a local run |

## Run

**Google Colab.** Open the notebook with the badge above, choose *Runtime → Change runtime type → T4 GPU* (the CPU runtime works, but is several times slower) and *Runtime → Run all*. The first cell clones this repository and unpacks the data; nothing has to be uploaded.

**Locally.** Python 3.9+:

```bash
git clone https://github.com/d9nk/deeplob-project.git && cd deeplob-project
pip install -r requirements.txt
jupyter notebook DeepLOB.ipynb
```

`Run all` takes about 12 minutes on a laptop CPU. It trains every model on a subset as an end-to-end check of the code and rebuilds every reported table and figure from `results/full_runs.zip`. The reported numbers themselves come from full-length runs of the same code (≈116 GPU-hours on an RTX 3090); the notebook shows the one-line call that repeats any of them.

## References

- Zhang, Z., Zohren, S., Roberts, S. (2019). DeepLOB: Deep Convolutional Neural Networks for Limit Order Books. IEEE Transactions on Signal Processing, 67(11). arXiv:1808.03668. Code: github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
- Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., Iosifidis, A. (2018). Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods. Journal of Forecasting, 37(8). [Dataset](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649), CC BY 4.0.
- Zhang, Z., Zohren, S. (2021). Multi-horizon forecasting for limit order books. arXiv:2105.10430.
- Prata, M. et al. (2023). LOB-based deep learning models for stock price trend prediction: a benchmark study (LOBCAST). arXiv:2308.01915.
