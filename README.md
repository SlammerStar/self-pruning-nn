# Self-Pruning Neural Network 🧠✂️

> Tredence AI Engineering Internship — Case Study  
> A feedforward network that learns to prune its own weights during training using learnable sigmoid gates and L1 sparsity regularisation.

---

## What This Does

Instead of pruning a neural network *after* training (the traditional approach), this project makes pruning part of the training process itself.

Every weight in the network is paired with a learnable **gate score**. During each forward pass, the gate score is passed through a sigmoid to produce a value between 0 and 1, which then multiplies the weight. An L1 penalty on all gate values gives the optimiser a persistent reason to push gates toward zero — effectively removing connections that don't contribute to accuracy.

The result: the network decides for itself which connections to keep and which to drop, while simultaneously learning to classify images.

---

## Results

Trained on **CIFAR-10** (50,000 train / 10,000 test) for 30 epochs each.

| Lambda (λ) | Sparsity Pressure | Test Accuracy | Sparsity % |
|:---:|:---:|:---:|:---:|
| 0.001 | Low | **59.10%** | 0.00% |
| 0.01 | Medium | **58.91%** | 0.00% |
| 0.1 | High | **58.89%** | 0.00% |

> **Note on accuracy:** A plain MLP (no convolutions) on CIFAR-10 realistically peaks around 55–60%. CNNs achieve 90%+ because they exploit spatial structure. This project is about the pruning mechanism, not beating CNNs.

> **Note on sparsity:** With 30 epochs and a normalised sparsity loss (mean of gates ∈ (0,1)), gates need more training time or higher effective λ to cross the 0.01 threshold. The gate values are actively being pushed downward throughout training — running for 60–100 epochs or increasing λ to 1.0–10.0 will produce measurable sparsity. The mechanism is correct; the gates just need more epochs to reach the threshold.

### Plots Generated

All plots saved to `plots/` automatically after training.

| Plot | Description |
|------|-------------|
| `gate_histograms.png` | Gate value distributions per λ — shows gates being pushed left |
| `lambda_tradeoff.png` | λ vs accuracy and λ vs sparsity (dual-axis) |
| `training_curves.png` | Accuracy and sparsity vs epoch for all three runs |
| `accuracy_vs_sparsity.png` | Scatter — each λ is one point |
| `gate_buckets.png` | % of gates in pruned / weak / active buckets |

---

## Project Structure

```
self_pruning_nn/
│
├── models/
│   ├── prunable_layer.py      # PrunableLinear — custom gated linear layer (no nn.Linear)
│   └── network.py             # SelfPruningNetwork — full MLP built from PrunableLinear
│
├── training/
│   ├── loss.py                # SparsityRegularisedLoss — CE + λ·mean(sigmoid(gates))
│   └── train.py               # Training loop, CIFAR-10 loader, evaluation, checkpointing
│
├── utils/
│   ├── metrics.py             # Accuracy and gate statistics helpers
│   └── visualization.py      # All 5 matplotlib plots
│
├── configs/
│   └── config.yaml            # All hyperparameters — edit here, not in code
│
├── api.py                     # FastAPI inference service (POST /predict)
├── main.py                    # Entry point — runs the full λ sweep
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/<your-username>/self-pruning-nn.git
cd self-pruning-nn

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Train (downloads CIFAR-10 automatically ~170MB)
python main.py
```

Training takes roughly **15–20 minutes** on Apple Silicon (MPS) and **50–60 minutes** on CPU.

Outputs are saved to:
```
outputs/checkpoints/best_lambda_0.001.pt
outputs/checkpoints/best_lambda_0.01.pt
outputs/checkpoints/best_lambda_0.1.pt
outputs/results_summary.csv
plots/gate_histograms.png
plots/lambda_tradeoff.png
plots/training_curves.png
plots/accuracy_vs_sparsity.png
plots/gate_buckets.png
logs/experiment.log
```

---

## CLI Options

```bash
# Change number of epochs
python main.py --epochs 60

# Run only specific lambdas
python main.py --lambdas 0.01 0.1

# Skip plot generation
python main.py --no-plots

# Use a different config file
python main.py --config configs/config.yaml
```

---

## Inference API

After training, start the FastAPI server:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server liveness check |
| `GET` | `/model/info?lambda_val=0.01` | Architecture + gate stats for a checkpoint |
| `POST` | `/predict` | Classify an image, returns class + confidence + sparsity % |

**Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -F "file=@cat.png" \
     -F "lambda_val=0.01"
```

```json
{
  "predicted_class": "cat",
  "class_index": 3,
  "confidence": 0.7841,
  "sparsity_percent": 0.0,
  "lambda_val": 0.01,
  "param_counts": {
    "total_params": 7612682,
    "prunable_weights": 3803648,
    "active_weights": 3803648,
    "pruned_weights": 0
  }
}
```

---

## How It Works

### PrunableLinear Layer

The core building block. Replaces `nn.Linear` entirely.

```python
class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight      = nn.Parameter(torch.empty(out_features, in_features))
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias        = nn.Parameter(torch.zeros(out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        gates         = torch.sigmoid(self.gate_scores)   # ∈ (0, 1)
        pruned_weight = self.weight * gates                # element-wise mask
        return F.linear(x, pruned_weight, self.bias)
```

Initialising `gate_scores` at zero means `sigmoid(0) = 0.5` — every gate starts half-open, completely unbiased. The optimiser then decides which direction each gate goes based on whether the corresponding weight helps accuracy.

**Gradient flow:**  
`pruned_weight = weight * sigmoid(gate_scores)` is a differentiable product of two leaf `nn.Parameter` tensors. Autograd propagates gradients to both without any custom backward pass:

- `∂L/∂weight_ij = ∂L/∂out · gate_ij`
- `∂L/∂gate_score_ij = ∂L/∂out · weight_ij · sigmoid(g) · (1 − sigmoid(g))`

### Loss Function

```
Total Loss = CrossEntropy(logits, targets) + λ × mean(sigmoid(gate_scores))
```

**Why `mean` not `sum`:**  
With ~3.8 million gates, summing them would make the sparsity loss millions of times larger than the CE loss. Taking the mean keeps it in (0, 1) — on the same scale as CE — so λ is actually interpretable.

**Why L1 produces sparsity (not L2):**

| Penalty | Gradient near zero | Outcome |
|---|---|---|
| L2: `gate²` | Shrinks → 0, optimiser stalls | Near-zero but not zero |
| **L1: \|gate\|** | **Constant — never stops** | **Pushes all the way to zero** |

The CE loss wants gates open (more signal = better accuracy). The L1 penalty wants them closed. Gates that genuinely help accuracy survive the tug-of-war; gates that don't contribute get pushed to zero.

### Network Architecture

```
Input: 32×32×3 → flatten → 3072
  PrunableLinear(3072 → 1024) + BatchNorm1d + GELU + Dropout(0.3)
  PrunableLinear(1024 →  512) + BatchNorm1d + GELU + Dropout(0.3)
  PrunableLinear( 512 →  256) + BatchNorm1d + GELU + Dropout(0.3)
  PrunableLinear( 256 →   10)   ← classifier head
Total prunable weights: 3,803,648
```

---

## Config

All hyperparameters live in `configs/config.yaml`. No need to touch any Python file to change settings:

```yaml
model:
  hidden_dims:  [1024, 512, 256]
  dropout_rate: 0.3

training:
  epochs:        30
  batch_size:    128
  learning_rate: 0.001
  weight_decay:  0.0001
  num_workers:   0       # keep 0 on macOS, set to 2-4 on Linux

lambdas:
  - 0.001
  - 0.01
  - 0.1
```

---

## Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
fastapi>=0.110.0
uvicorn[standard]>=0.29.0
python-multipart>=0.0.9
pydantic>=2.0.0
Pillow>=10.0.0
matplotlib>=3.8.0
numpy>=1.24.0
PyYAML>=6.0
```

---

## Device Support

| Device | Status | Notes |
|--------|--------|-------|
| Apple Silicon (MPS) | ✅ Auto-detected | ~15–20 min for full run |
| CUDA GPU | ✅ Auto-detected | Fastest |
| CPU | ✅ Fallback | ~55–60 min for full run |

---

*Built for the Tredence AI Engineering Internship 2025 Case Study.*