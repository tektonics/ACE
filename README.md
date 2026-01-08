# ACE-Code: Automated Circuit Extraction for Code Models

## Phase 1: SAE-Based Anomaly Flagging

A **negative restraint system** that acts as a high-speed error alarm by detecting features correlated with bugs rather than certifying correctness.

### Overview

This implementation leverages Sparse Autoencoders (SAEs) to detect anomalous patterns in code generation. By analyzing the residual stream of a smaller "sidecar" model, we identify SAE features that correlate with incorrect code generation.

**Key Finding:** Models possess robust "anomaly detectors" (incorrectness features, F1 ≈ 0.821) but lack reliable "validity assessors" (correctness features, F1 ≈ 0.50). We exploit this asymmetry.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Production Model                     │
│                    (e.g., Llama-70B)                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Input Prompt
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Sidecar Model                             │
│              (Llama-3.1-8B-Instruct)                        │
│                                                              │
│  ┌──────────────────┐    ┌─────────────────────────────┐   │
│  │  Residual Stream │───▶│     JumpReLU SAE            │   │
│  │   (Layer 12)     │    │  (8x expansion, ~32k feat)  │   │
│  └──────────────────┘    └─────────────────────────────┘   │
│                                        │                    │
│                                        ▼                    │
│                          ┌─────────────────────────────┐   │
│                          │ Incorrectness Features      │   │
│                          │ (top 100 by t-statistic)    │   │
│                          └─────────────────────────────┘   │
│                                        │                    │
│                                        ▼                    │
│                          ┌─────────────────────────────┐   │
│                          │   Anomaly Score             │   │
│                          │ Σ(activation × weight)      │   │
│                          └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
              Score > Threshold    Score ≤ Threshold
                    │                   │
                    ▼                   ▼
         ┌──────────────────┐  ┌──────────────────┐
         │ Route to Phase 4 │  │  Proceed Normal  │
         │ (Formal Verify)  │  │                  │
         └──────────────────┘  └──────────────────┘
```

### Mathematical Foundation

#### JumpReLU SAE

Given residual stream activation $x \in \mathbb{R}^d$:

```
Encoder: a(x) = JumpReLU_θ(x @ W_enc + b_enc)
JumpReLU: JumpReLU_θ(z) = z ⊙ H(z - θ)
Decoder: x̂ = a @ W_dec + b_dec
```

Where:
- $H$ is the Heaviside step function
- $θ$ is a learnable threshold vector
- Expansion factor: 8x (4096 → 32,768 latents)

#### Feature Selection (t-statistic)

```
t^{incorrect}_{l,j} = (μ(incorrect) - μ(correct)) /
                      sqrt(σ²(incorrect)/N_incorrect + σ²(correct)/N_correct)
```

Features with highest positive t-statistics for the "incorrect" class are selected as incorrectness indicators.

### Installation

```bash
# Clone the repository
git clone https://github.com/tektonics/ACE.git
cd ACE

# Install dependencies
pip install -e .

# Or install requirements directly
pip install -r requirements.txt
```

### Quick Start

#### 1. Feature Discovery (Offline)

```python
from ace_code.sae_anomaly import SAEAnomalyPipeline

# Create and run discovery pipeline
pipeline = SAEAnomalyPipeline()
pipeline.load_model()
result = pipeline.run_discovery()

# Save artifacts for production use
pipeline.save("artifacts/")
```

Or use the CLI:

```bash
python scripts/run_discovery.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --layer 12 \
    --top-k 100 \
    --output-dir artifacts/
```

#### 2. Runtime Detection (Online)

```python
from ace_code.sae_anomaly import SAEAnomalyPipeline

# Load trained pipeline
pipeline = SAEAnomalyPipeline.load("artifacts/")

# Detect anomalies
result = pipeline.detect("def sort_list(lst): return lst.sort()")

if result.is_anomaly:
    print(f"⚠️ Anomaly detected! Score: {result.anomaly_score:.3f}")
    # Route to Phase 4 formal verification
else:
    print("✓ No anomaly detected")
```

Or use the CLI:

```bash
python scripts/run_detection.py \
    --artifacts artifacts/ \
    --input "def sort_list(lst): return lst.sort()"
```

### Configuration

Create a configuration file (`configs/default.yaml`):

```yaml
# Model configuration
model_name: "meta-llama/Llama-3.1-8B-Instruct"
target_layer: 12
device: "auto"
quantization: "8bit"

# SAE configuration
sae_expansion_factor: 8

# Discovery configuration
dataset_name: "mbpp"
top_k_features: 100
min_t_statistic: 2.0
max_frequency: 0.02

# Detection configuration
anomaly_threshold: 0.5
```

### API Reference

#### `JumpReLUSAE`

```python
from ace_code.sae_anomaly import JumpReLUSAE

sae = JumpReLUSAE(
    d_model=4096,        # Model hidden dimension
    expansion_factor=8,  # Latent expansion (4096 → 32k)
    device="cuda",
)

# Encode to sparse features
features = sae.encode(residual_stream)

# Full forward pass
x_hat, features = sae(x, return_features=True)
```

#### `SidecarModel`

```python
from ace_code.sae_anomaly import SidecarModel, SidecarConfig

config = SidecarConfig(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    target_layer=12,
    quantization="8bit",
)

sidecar = SidecarModel(config).load()

# Get SAE features for input
features = sidecar.get_sae_features("def hello():", position="final")
```

#### `FeatureDiscovery`

```python
from ace_code.sae_anomaly import FeatureDiscovery, DiscoveryConfig

config = DiscoveryConfig(
    dataset_name="mbpp",
    top_k_features=100,
)

discovery = FeatureDiscovery(sidecar, config)
result = discovery.discover()

print(f"Found {result.n_selected_features} incorrectness features")
print(f"Top t-statistics: {result.feature_t_statistics[:5]}")
```

#### `AnomalyDetector`

```python
from ace_code.sae_anomaly import AnomalyDetector, DetectorConfig

detector = AnomalyDetector(sidecar, discovery_result, DetectorConfig())

result = detector.detect("def buggy_code():")
print(f"Anomaly: {result.is_anomaly}, Score: {result.anomaly_score}")
```

### Model Support

| Model | Target Layer | Hidden Dim | Status |
|-------|-------------|------------|--------|
| Llama-3.1-8B-Instruct | 12 | 4096 | ✓ Recommended |
| Gemma-2-2B-IT | 19 | 2304 | ✓ Lightweight |

### Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=ace_code --cov-report=html

# Run specific test file
pytest tests/test_sae.py -v
```

### References

This implementation is based on:

1. **Tahimic & Cheng (2025)**: "Mechanistic Interpretability of Code Correctness in LLMs via Sparse Autoencoders"
   - Core math for feature selection (t-statistics)
   - Discovery of "incorrectness asymmetry"
   - Prompt templates

2. **Nguyen et al. (2025)**: "Deploying Interpretability to Production with Rakuten: SAE Probes for PII Detection"
   - "Sidecar" architecture
   - Production deployment patterns

3. **Lieberum et al. (2024)**: "Gemma Scope: Open Sparse Autoencoders Everywhere All at Once on Gemma 2"
   - JumpReLU SAE architecture
   - Training methodology

### Expected Performance

- **Incorrectness Features**: F1 ≈ 0.821
- **Correctness Features**: F1 ≈ 0.50 (unreliable, not used)

### Checklist

- [x] Model: Llama-3.1-8B-Instruct (Sidecar)
- [x] Extraction: Layer 12, Final Prompt Token
- [x] SAE: JumpReLU, Expansion Factor 8
- [x] Selection: T-stats on MBPP dataset; keep top "incorrect" features
- [x] Deployment: Feature Activation > Threshold → Route to Phase 4

### License

Apache 2.0
