# QuickSort

QuickSort is an extreme classification tool that fine-tunes image embeddings using human-in-the-loop reinforcement learning.

<img width="1462" height="735" alt="image" src="https://github.com/user-attachments/assets/2dc08bb8-66d1-42f9-bf6c-522f6b690843" />

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
