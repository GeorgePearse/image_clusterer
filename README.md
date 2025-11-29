# QuickSort

QuickSort is an extreme classification tool that fine-tunes image embeddings using human-in-the-loop reinforcement learning.

## Features

- **Synthetic Data Generation**: Creates a test dataset of images.
- **Interactive UI**: Simple web interface for "Same/Different" voting.
- **Active Learning**: Uses Qdrant to find hard negatives/positives for efficient training.
- **Real-time Fine-tuning**: Updates the embedding model on-the-fly based on user feedback.

## Documentation

Full documentation is available in the `docs/` directory. To serve it locally:

```bash
mkdocs serve
```

## Quick Start

```bash
./start.sh
```
