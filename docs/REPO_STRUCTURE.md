# REPO_STRUCTURE.md

The repository must follow this structure:

src/
  data/
    dataset.py
    preprocessing.py
    loader.py
    lunarcrush_api.py

  models/
    real_lstm.py
    real_lstm_attention.py
    attention.py
    quaternion_ops.py
    quaternion_lstm.py
    qnn_attention_model.py
    hierarchical_qlstm.py
    revin.py
    dish_ts.py

  training/
    trainer.py
    losses.py

  evaluation/
    metrics.py
    directional_accuracy.py
    sharpe_ratio.py

configs/
  data/
    daily/                           # Daily data configs per asset
    hourly/                          # Hourly data configs
    4hourly/                         # 4-hourly data configs (including btc_hier.yaml)
  experiments/
    full_comparison.yaml             # 7-variant baseline comparison
    hourly_hierarchical.yaml         # 6-variant hierarchical QLSTM (hourly)
    4hourly_hierarchical.yaml        # 6-variant hierarchical QLSTM (4-hourly)

experiments/
  run_experiments.py
  visualize_results.py

notebooks/
  Hierarchical_QLSTM_BTC_Hourly.ipynb
  4Hourly_Hierarchical_QLSTM_BTC.ipynb
