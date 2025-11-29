# Simulation Guide

The simulation framework allows you to test and compare selection strategies without manual labeling. This is essential for:

- **Strategy Development**: Quickly iterate on new selection algorithms
- **Performance Benchmarking**: Compare strategies objectively
- **Research**: Generate reproducible results for papers
- **Cost Estimation**: Predict labeling time/cost before hiring annotators

## How It Works

The simulation uses an **Oracle** - a perfect labeler with access to ground truth. The Oracle simulates a human annotator, allowing you to run the full labeling loop automatically.

### Flow

```
1. Strategy selects an image
   ↓
2. Oracle provides ground truth label
   ↓
3. Label is added to dataset.user_labels
   ↓
4. Metrics are calculated (accuracy, purity, etc.)
   ↓
5. Repeat until all images labeled or budget exhausted
```

## Local Simulation

### Basic Usage

Run all strategies and compare:

```bash
python -m backend.simulate
```

Run a specific strategy:

```bash
python -m backend.simulate --strategy cluster_chain
```

### Advanced Options

```bash
python -m backend.simulate \
    --strategy uncertainty \
    --num-images 5000 \
    --max-labels 500 \
    --log-every 25 \
    --output results.json
```

**Parameters:**
- `--strategy`: Which strategy to test (or `all` for comparison)
- `--num-images`: Dataset size (default: 1000)
- `--max-labels`: Stop after N labels (default: label all images)
- `--log-every`: Metric logging frequency (default: 10)
- `--output`: Save results to JSON file

### Example Output

```
============================================================
Initializing simulation: cluster_chain
============================================================
Loading 1000 images...
Initializing model...
Initializing Qdrant (in-memory)...
Computing embeddings and indexing...
Initialization complete!

Step    1 | Labeled:    1 | Accuracy: 1.000 | Purity: 1.000 | Select: 45.2ms
Step   10 | Labeled:   10 | Accuracy: 1.000 | Purity: 1.000 | Select: 12.3ms
Step   20 | Labeled:   20 | Accuracy: 1.000 | Purity: 1.000 | Select: 11.8ms
...
Step 1000 | Labeled: 1000 | Accuracy: 1.000 | Purity: 1.000 | Select: 10.5ms

============================================================
Simulation Complete: cluster_chain
============================================================
Total time: 45.23s
Labels collected: 1000
Final accuracy: 1.000
Final purity: 1.000
Avg selection time: 11.2ms
```

## Remote Simulation

For larger experiments or parallel execution, run simulations on remote compute.

### E2B (Recommended for CPU)

E2B provides sandboxed Python environments - perfect for CPU-based simulations.

**Setup:**

```bash
# Install E2B
pip install e2b-code-interpreter

# Get API key from https://e2b.dev
export E2B_API_KEY="your_api_key"
```

**Run single strategy:**

```bash
python -m backend.simulate_remote --platform e2b --strategy cluster_chain
```

**Run all strategies in parallel:**

```bash
python -m backend.simulate_remote --platform e2b --parallel
```

This spawns multiple sandboxes simultaneously, one per strategy.

### Modal (Alternative)

Modal provides serverless compute with easy Python deployment.

**Setup:**

```bash
# Install Modal
pip install modal

# Authenticate
modal token new
```

**Run simulation:**

```bash
python -m backend.simulate_remote --platform modal --strategy uncertainty
```

Modal is great if you want GPU acceleration or need more compute resources.

### Platform Comparison

| Feature | E2B | Modal |
|---------|-----|-------|
| **Isolation** | Full sandbox | Container |
| **Parallel Execution** | Easy (ThreadPool) | Built-in |
| **CPU Performance** | Good | Good |
| **GPU Support** | No | Yes |
| **Cost** | Pay per second | Pay per second |
| **Best For** | CPU simulations, security | GPU training, scale |

For QuickSort simulations (Qdrant in-memory, CPU-only), **E2B is simpler and cheaper**.

## Understanding Metrics

### Accuracy

```
accuracy = correct_labels / total_labels
```

Percentage of user labels that match ground truth. In simulation with Oracle (noise_rate=0), this should always be 1.0.

### Cluster Purity

```
purity = average(class_accuracy for each class)
```

For each true class, what percentage of its labeled examples are correct? High purity means the strategy is labeling coherent clusters.

### Label Distribution

How many labels per class have been collected. Good strategies should have balanced distribution unless the dataset is imbalanced.

### Selection Time

Time to run `strategy.select_next()`. This measures computational efficiency:
- **Random**: ~0.1ms (instant)
- **Cluster Chain**: ~10-50ms (needs one vector search)
- **Uncertainty**: ~100-500ms (samples and evaluates multiple candidates)

## Creating Custom Strategies

Implement the `SelectionStrategy` interface:

```python
from backend.app.oracle import SelectionStrategy

class MyStrategy(SelectionStrategy):
    def __init__(self, **kwargs):
        super().__init__("my_strategy")
        # Your initialization
    
    def select_next(self, model, dataset, qdrant_client, 
                     collection_name, user_labels, last_labeled_id=None):
        # Your selection logic
        unlabeled = [id for id in dataset.ids if id not in user_labels]
        
        if not unlabeled:
            return None
        
        # Example: Random selection
        selected_id = np.random.choice(unlabeled)
        self.selection_history.append(selected_id)
        return selected_id
```

Register in `backend/app/oracle.py`:

```python
STRATEGIES["my_strategy"] = MyStrategy
```

Test it:

```bash
python -m backend.simulate --strategy my_strategy
```

## Real-World Simulation

For production planning, add noise to simulate human error:

```python
# In backend/simulate.py
oracle = Oracle(dataset, noise_rate=0.05)  # 5% error rate
```

Or model inter-annotator agreement:

```python
# Simulate multiple annotators voting
class MultiAnnotatorOracle:
    def __init__(self, dataset, num_annotators=3, agreement_rate=0.9):
        self.annotators = [
            Oracle(dataset, noise_rate=(1-agreement_rate))
            for _ in range(num_annotators)
        ]
    
    def label(self, image_id):
        votes = [a.label(image_id) for a in self.annotators]
        # Majority vote
        return Counter(votes).most_common(1)[0][0]
```

## Example Workflow

Research a new strategy:

```bash
# 1. Local quick test
python -m backend.simulate --strategy my_new_strategy --num-images 100

# 2. Full local benchmark
python -m backend.simulate --strategy all --num-images 1000 --output local_results.json

# 3. Large-scale remote test
python -m backend.simulate_remote \
    --platform e2b \
    --strategy my_new_strategy \
    --num-images 10000 \
    --max-labels 2000 \
    --output remote_results.json

# 4. Parallel comparison
python -m backend.simulate_remote --platform e2b --parallel --num-images 5000
```

## Interpreting Results

**Example comparison output:**

```
============================================================
STRATEGY COMPARISON
============================================================
Strategy             Labels     Accuracy     Time (s)    
------------------------------------------------------------
random               1000       1.000        42.15       
cluster_chain        1000       1.000        45.23       
uncertainty          1000       1.000        125.67      
margin               1000       1.000        118.45      
diversity            1000       1.000        98.33       
```

**Insights:**
- All reach 100% accuracy (Oracle has no noise)
- `cluster_chain` is nearly as fast as `random` despite complex logic
- `uncertainty` and `margin` are 2-3x slower due to sampling overhead
- For production, consider `cluster_chain` for speed or `uncertainty` for quality

## Cost Estimation

Use simulation to predict real labeling costs:

```python
# Assume: 
# - cluster_chain achieves 2000 labels/hour (from simulation)
# - Need to label 100K images
# - Annotator costs $20/hour

hours = 100_000 / 2000  # 50 hours
cost = hours * 20       # $1000

print(f"Estimated cost: ${cost} over {hours} hours")
```

Compare to random sampling at 200 labels/hour:

```python
hours_random = 100_000 / 200  # 500 hours
cost_random = hours_random * 20  # $10,000

savings = cost_random - cost  # $9,000 (90% reduction!)
```

## Reproducibility

For reproducible results, seed the random generators:

```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)
```

Or in simulation CLI:

```bash
PYTHONHASHSEED=42 python -m backend.simulate --strategy cluster_chain
```

## Troubleshooting

**Simulation hangs:**
- Check if Qdrant is running out of memory
- Reduce `--num-images`
- Use `use_qdrant_memory=True` in SimulationRunner

**Poor accuracy:**
- Verify Oracle noise_rate is set correctly
- Check dataset ground truth labels
- Ensure model is initialized (not training from random)

**Slow selection:**
- Profile with `--log-every 1` to see per-step timing
- Sample fewer candidates in uncertainty/margin strategies
- Use smaller `k_neighbors` parameter

## Next Steps

- Read [Architecture](architecture.md) to understand the system design
- See [Usage](usage.md) for manual labeling workflow
- Check [Roadmap](roadmap.md) for future simulation features
