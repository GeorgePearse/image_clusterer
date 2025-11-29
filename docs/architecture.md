# Architecture

## System Design

QuickSort moves beyond simple random sampling. It uses a **Neighborhood-Based Active Learning** strategy to maximize labelling throughput.

### Components

1.  **Frontend (React + Vite)**:
    *   **Blended Interface**: Single input field that handles both verification (Y/N) and correction.
    *   **regl-scatterplot**: WebGL-powered 2D visualization of the embedding space.
    *   **WebSocket Logs**: Real-time streaming of backend events.

2.  **Backend (FastAPI)**:
    *   **PyTorch Model**: A simple CNN (ResNet-like) that maps images to a 64-dimensional vector space.
    *   **Squeeze**: A custom dimensionality reduction library (UMAP) for generating the 2D projection.
    *   **Modal Ops**: Background training and re-indexing tasks (optional integration).

3.  **Vector Database (Qdrant)**:
    *   Stores image embeddings.
    *   Performs efficient K-Nearest Neighbor (KNN) searches to find context for the current image.

### The "Chain Reaction" Loop

The core innovation of QuickSort is the **Chain Reaction** loop:

1.  **User Labels Image A**: "Cat".
2.  **System Search**: Backend queries Qdrant for the nearest **unlabelled** neighbor of Image A. Let's call it Image B.
3.  **Context Retrieval**: Backend checks the neighbors of Image B. Since Image A is close (and now labelled "Cat"), "Cat" becomes the strong suggestion for Image B.
4.  **Presentation**: Frontend shows Image B with "Cat" as the suggestion.
5.  **User Confirmation**: User presses `Shift + Enter` (instant confirmation).
6.  **Repeat**: The system finds the neighbor of Image B, continuing the chain until the cluster is fully labelled.

### Data Flow

1.  **Initialization**:
    *   Images are embedded via the PyTorch model.
    *   Embeddings are stored in Qdrant.
    *   `Squeeze` computes the initial 2D projection.

2.  **Fetching Next Sample (`/next`)**:
    *   System checks `last_labeled_id`.
    *   Finds nearest unlabelled neighbor (Cluster Exploration) or falls back to random/uncertainty sampling.
    *   Retrieves label distribution of neighbors to generate a suggestion.

3.  **Labelling (`/label`)**:
    *   User submits label.
    *   Label is stored in memory/DB.
    *   Local 2D points are updated optimistically.
    *   (Optional) Model retrains in background on the new labeled pair.
