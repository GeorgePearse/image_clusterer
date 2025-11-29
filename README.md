# QuickSort

**Label 10,000 images in an afternoon.**

QuickSort is an extreme-speed image labeling tool that turns data annotation from a tedious chore into a flow state experience. Using active learning and the "Chain Reaction" strategy, you can achieve 10x faster labeling throughput compared to traditional random sampling.

## The Problem

Traditional image labeling is painfully slow:
- Random sampling wastes time on easy, redundant examples
- No context between consecutive labels
- Constant context switching breaks focus
- No autocomplete or suggestions from existing labels

**Result:** Labeling 10,000 images takes weeks of mind-numbing work.

## The Solution: Chain Reaction

QuickSort implements a novel **Cluster Chaining** strategy:

1. You label Image A as "cat"
2. The system finds the *nearest unlabeled neighbor* to Image A
3. Because they're similar, it suggests "cat"
4. You press `J` to confirm in 200ms
5. The chain continues through the entire cluster

This creates a **flow state** where you're rapidly confirming correct suggestions instead of manually typing labels. When you hit a cluster boundary, you correct the label and start a new chain.

### Speed Comparison

| Strategy | Images/Hour | Time for 10K |
|----------|-------------|--------------|
| Random Sampling | ~200 | 50 hours |
| QuickSort (Chain) | ~2000+ | 5 hours |

<img width="1462" height="735" alt="image" src="https://github.com/user-attachments/assets/2dc08bb8-66d1-42f9-bf6c-522f6b690843" />

## Features

### Interactive Labeling UI
- **Keyboard-First Design**: `J` to accept, `I` to correct, `Alt+Arrow` for history
- **Smart Autocomplete**: Learns from your labels in real-time
- **WebGL Scatter Plot**: See your labeled clusters emerge in 2D space
- **Real-time Logs**: Watch the system learn via WebSocket stream

### Active Learning Engine
- **5 Selection Strategies**: Random, Cluster Chain, Uncertainty, Margin, Diversity
- **Qdrant Vector Search**: Find similar images in milliseconds
- **Live Model Updates**: Fine-tune embeddings based on your feedback
- **Modal Integration**: Offload training to serverless GPUs

### Simulation Framework
- **Oracle System**: Test strategies without human labeling
- **Automated Benchmarking**: Compare selection strategies objectively
- **Remote Execution**: Run simulations on E2B or Modal
- **Detailed Metrics**: Track accuracy, purity, and labeling efficiency

## Quick Start

```bash
./start.sh
```

Open `http://localhost:5173` and start labeling!

### Basic Workflow

1. **Bootstrap Phase**: Type a few initial labels to teach the system
2. **Flow Phase**: Press `J` to accept suggestions and chain through clusters
3. **Correction**: Press `I` when suggestion is wrong, type correct label
4. **Repeat**: Watch the scatter plot fill with colored clusters

See [`docs/usage.md`](docs/usage.md) for full keyboard shortcuts and tips.

## Simulation & Benchmarking

Compare selection strategies without manual labeling:

```bash
# Run all strategies locally
python -m backend.simulate --num-images 1000

# Run specific strategy
python -m backend.simulate --strategy cluster_chain --max-labels 100

# Run on E2B (remote sandbox)
python -m backend.simulate_remote --platform e2b --parallel

# Run on Modal (serverless)
python -m backend.simulate_remote --platform modal --strategy uncertainty
```

### Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| `cluster_chain` | Follow nearest neighbors (default) | High-density clusters |
| `random` | Random sampling baseline | Uniform coverage |
| `uncertainty` | Query most ambiguous images | Hard examples |
| `margin` | Find cluster boundaries | Rare classes |
| `diversity` | Maximize embedding space coverage | Exploration |

Extend the system by adding your own strategy in [`backend/app/oracle.py`](backend/app/oracle.py:245).

## Architecture

```
┌─────────────────┐
│  React Frontend │  WebGL scatter, keyboard shortcuts
└────────┬────────┘
         │ WebSocket + REST
┌────────┴────────┐
│  FastAPI Backend│  Selection strategies, embedding model
└────────┬────────┘
         │
    ┌────┴─────┬───────────┐
    ▼          ▼           ▼
┌────────┐ ┌────────┐ ┌────────┐
│ Qdrant │ │PyTorch │ │ Modal  │
│ Vector │ │  CNN   │ │Training│
│ Search │ │Embedding│ │  GPU   │
└────────┘ └────────┘ └────────┘
```

**Key Components:**
- **PyTorch CNN**: Maps images to 64D normalized embeddings
- **Qdrant**: Vector database for KNN search (cosine similarity)
- **Squeeze**: Fast UMAP implementation for 2D projection
- **Modal**: Serverless GPU training (optional)

See [`docs/architecture.md`](docs/architecture.md) for details.

## Documentation

Full docs available in the `docs/` directory:
- [Architecture](docs/architecture.md) - System design and data flow
- [Usage Guide](docs/usage.md) - Keyboard shortcuts and workflow
- [API Reference](docs/api.md) - REST endpoints
- [Roadmap](docs/roadmap.md) - Future plans (VLMs, CLIP, etc.)

Serve locally with:

```bash
mkdocs serve
```

## Configuration

Control behavior via environment variables:

```bash
# Change selection strategy
export SELECTION_STRATEGY=uncertainty

# Adjust dataset size
export NUM_IMAGES=5000

# Enable update on every vote (slower but more reactive)
export UPDATE_ON_EVERY_VOTE=true

./start.sh
```

## Extending QuickSort

### Add a Custom Selection Strategy

```python
# backend/app/oracle.py

class MyCustomStrategy(SelectionStrategy):
    def select_next(self, model, dataset, qdrant_client, collection_name, 
                     user_labels, last_labeled_id=None):
        # Your selection logic here
        return selected_image_id

# Register it
STRATEGIES["my_strategy"] = MyCustomStrategy
```

Then use it: `export SELECTION_STRATEGY=my_strategy`

### Add a Custom Dataset

Replace `MNISTWrapper` in `backend/app/common.py` with your own dataset class implementing:
- `get_tensor(idx)` - Return image tensor
- `get_base64(idx)` - Return base64-encoded PNG
- `ids` - List of unique image IDs
- `labels` - Ground truth labels (for simulation)

## Why "QuickSort"?

Like the sorting algorithm, QuickSort is fast because it exploits structure. Traditional labeling is O(n) with no shortcuts. QuickSort uses embeddings to partition the space and chain through clusters, achieving better constants and cache-friendly access patterns.

Plus, it's a pun on "quick classification" and the satisfying feeling of rapidly sorting through images.

## Performance Tips

1. **Use Cluster Chain First**: Label one image per cluster, then chain through each
2. **Trust Suggestions**: The model learns fast - don't second-guess obvious confirmations
3. **Keyboard Only**: Avoid touching the mouse to maintain flow
4. **Batch Sessions**: 30-minute focused sessions beat hours of distracted work

## Contributing

PRs welcome! Areas of interest:
- New selection strategies
- Dataset adapters (ImageNet, COCO, custom)
- Alternative embedding models (CLIP, DINOv2, ResNet)
- Better 2D projection algorithms
- UI improvements

## License

MIT

## Citation

If you use QuickSort in research, please cite:

```bibtex
@software{quicksort2024,
  title={QuickSort: Flow-State Image Labeling via Cluster Chaining},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/quicksort}
}
```
