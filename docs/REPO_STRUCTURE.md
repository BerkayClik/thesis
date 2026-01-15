# REPO_STRUCTURE.md

The repository must follow this structure:

src/
  data/
    dataset.py
    preprocessing.py

  models/
    real_lstm.py
    attention.py
    quaternion_ops.py
    quaternion_lstm.py
    qnn_attention_model.py

  training/
    trainer.py
    losses.py

  evaluation/
    metrics.py
    directional_accuracy.py

configs/
  base.yaml
  experiment.yaml

experiments/
  run_experiments.py
