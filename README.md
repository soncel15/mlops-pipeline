# MLOps Pipeline 🔧

End-to-end machine learning operations pipeline for production ML systems.

## Components
- **Data Pipeline** — data loading, validation, versioning
- **Training Pipeline** — configurable training with experiment tracking
- **Model Registry** — versioned model storage with metadata
- **Evaluation** — automated model evaluation and comparison
- **Serving** — model deployment with health monitoring

## Architecture
```
pipeline/
├── data/
│   ├── loader.py        # Dataset loading (local, S3, HuggingFace)
│   ├── validator.py     # Data quality checks
│   └── versioning.py    # Data version management
├── training/
│   ├── trainer.py       # Core training loop
│   ├── callbacks.py     # Early stopping, checkpointing, logging
│   └── distributed.py   # Multi-GPU training utilities
├── evaluation/
│   ├── evaluator.py     # Model evaluation suite
│   └── comparator.py    # Model comparison (A/B, champion/challenger)
├── registry/
│   ├── model_store.py   # Model artifact storage
│   └── metadata.py      # Experiment metadata tracking
└── config.py            # Pipeline configuration
```

## Usage
```python
from pipeline.training import Trainer
from pipeline.data import DataLoader
from pipeline.registry import ModelRegistry

# Configure
config = {
    "model": "resnet50",
    "dataset": "imagenet",
    "epochs": 100,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "optimizer": "adamw",
}

# Train
trainer = Trainer(config)
metrics = trainer.train()

# Register
registry = ModelRegistry()
registry.register("resnet50", "v3", trainer.model, metrics)
```
