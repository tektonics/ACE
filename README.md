# ACE-Code: Automated Circuit Extraction for Code Models

A mechanistic interpretability toolkit for understanding and steering code-capable language models through circuit discovery, sparse autoencoders, and activation steering.

## Overview

ACE-Code provides a complete pipeline for:

1. **Circuit Discovery (CD-T)**: Find which attention heads and layers are responsible for specific code predictions
2. **Error Detection (SAE)**: Identify "incorrectness" features using Sparse Autoencoders
3. **Intervention**: Fix bugs via inference-time steering or permanent weight editing (PISCES)

## Installation

```bash
# Clone the repository
git clone https://github.com/tektonics/ACE.git
cd ACE

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Quick Start

```python
from ace_code import ACEPipeline

# Initialize pipeline with a code model
pipeline = ACEPipeline("gemma-2-2b")
pipeline.load()

# Analyze code snippets
code_snippets = [
    '''def calculate_total(price: float, quantity: int) -> float:
        return price * quantity
    ''',
    '''def find_max(numbers: list[int]) -> int:
        max_val = numbers[0]
        for num in numbers:
            if num > max_val:
                max_val = num
        return max_val
    ''',
]

results = pipeline.full_analysis(code_snippets)
print(results.summary())
```

## Architecture

### Phase 1: Model Loading with Hooks

Uses TransformerLens to load models with hook access to internal activations:

```python
from ace_code import load_hooked_model

model = load_hooked_model("gemma-2-2b", device="cuda")
logits, cache = model.run_with_cache(input_ids)
resid = cache["blocks.12.hook_resid_post"]
```

### Phase 2: Mutation Testing

Generate positive/negative code pairs for contrastive analysis:

```python
from ace_code.data.mutation import MutationGenerator

generator = MutationGenerator()
pairs = generator.generate_pairs(code_snippets)

for pair in pairs:
    print(f"Original: {pair.positive}")
    print(f"Mutated: {pair.negative}")
    print(f"Mutation: {pair.mutation_type}")
```

### Phase 3: Circuit Discovery (CD-T)

Contextual Decomposition for Transformers finds relevant components:

```python
from ace_code.discovery import ContextualDecomposition

cdt = ContextualDecomposition(model)
result = cdt.decompose(input_ids, relevant_positions=[0, 1, 2])

print(f"Top layers: {result.get_top_layers(k=5)}")
print(f"Top heads: {result.get_top_heads(k=10)}")
```

### Phase 4: Error Detection (SAE)

Identify error-indicating features using Sparse Autoencoders:

```python
from ace_code.detection import SAEDetector

detector = SAEDetector(model, layer=12)
detector.load_sae("path/to/sae.pt")

error_features = detector.identify_error_features(
    positive_inputs=correct_code_tokens,
    negative_inputs=buggy_code_tokens,
)

for feature in error_features[:5]:
    print(f"Feature {feature.feature_idx}: t={feature.t_statistic:.2f}")
```

### Phase 5: Intervention

#### Inference-Time Steering

```python
from ace_code.intervention import SteeringVector

sv = SteeringVector.from_prompts(
    model,
    positive_prompt=correct_code,
    negative_prompt=buggy_code,
    layer=12,
)

# Apply steering during generation
output = sv.apply(model, input_ids, strength=1.5)
```

#### Permanent Weight Editing (PISCES)

```python
from ace_code.intervention import PISCES

pisces = PISCES(model)
result = pisces.remove_direction(
    error_direction,
    layer=12,
    component="mlp_out",
)
```

## Configuration

See `configs/default.yaml` for all configuration options:

```yaml
model:
  name: "gemma-2-2b"
  device: "cuda"
  dtype: "float16"

circuit_discovery:
  target_layers: [10, 11, 12, 13, 14]
  use_attention_shortcut: true

sae:
  layer: 12
  t_threshold: 2.0
  top_k_features: 20

steering:
  default_strength: 1.0
  normalize: true
```

## Project Structure

```
ACE/
├── ace_code/
│   ├── __init__.py          # Main exports
│   ├── pipeline.py          # ACEPipeline orchestrator
│   ├── core/
│   │   ├── model.py         # Model loading with hooks
│   │   └── activations.py   # Activation caching
│   ├── data/
│   │   ├── mutation.py      # Code mutation generators
│   │   └── datasets.py      # Dataset loading utilities
│   ├── discovery/
│   │   ├── cdt.py           # Contextual Decomposition
│   │   └── pahq.py          # Per Attention Head Quantization
│   ├── detection/
│   │   ├── sae_detector.py  # SAE-based error detection
│   │   └── feature_analysis.py
│   ├── intervention/
│   │   ├── steering.py      # Inference-time steering
│   │   └── pisces.py        # Permanent weight editing
│   └── utils/
│       ├── logging.py
│       ├── device.py
│       └── statistics.py
├── tests/
├── configs/
├── scripts/
├── requirements.txt
└── pyproject.toml
```

## Key Dependencies

- `torch>=2.0.0` - Deep learning framework
- `transformer_lens>=1.0.0` - Mechanistic interpretability hooks
- `sae_lens>=1.0.0` - Sparse Autoencoder loading/analysis
- `transformers>=4.30.0` - Model loading
- `tree-sitter>=0.20.0` - Code parsing for mutations

## Memory Optimization (PAHQ)

For large models (>7B parameters), use Per Attention Head Quantization:

```python
from ace_code.discovery import PAHQQuantizer, QuantizationConfig

config = QuantizationConfig(
    target_precision=torch.float32,
    background_precision=torch.float16,
)

pahq = PAHQQuantizer(model, config)
pahq.quantize_background()

# Analyze specific head with full precision
with pahq.focus_on_head(layer=10, head=5):
    result = cdt.decompose(input_ids)
```

## Running Tests

```bash
pytest tests/ -v
```

## Citation

If you use ACE-Code in your research, please cite:

```bibtex
@software{ace_code2024,
  title = {ACE-Code: Automated Circuit Extraction for Code Models},
  year = {2024},
  url = {https://github.com/tektonics/ACE}
}
```

## License

MIT License - see LICENSE file for details.
