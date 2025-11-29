# QuickSort

Welcome to QuickSort, an extreme classification system for image sorting.

## Overview

QuickSort uses a human-in-the-loop approach to fine-tune image embeddings using Reinforcement Learning (RL) and Qdrant's vector search capabilities.

The system consists of:
- **Backend**: FastAPI server managing the embedding model and Qdrant interactions.
- **Frontend**: A simple UI for displaying image pairs and capturing user feedback.
- **Qdrant**: Vector database for storing and retrieving image embeddings.

## Getting Started

To start the application, simply run:

```bash
./start.sh
```

This will launch the backend server and serve the frontend at `http://localhost:8000`.
