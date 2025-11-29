# Architecture

## Overview

QuickSort is a human-in-the-loop image labelling system that uses active learning to maximize labelling efficiency. The system fine-tunes image embeddings in real-time based on user feedback, using a "chain reaction" strategy to rapidly label clusters of similar images.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              QUICKSORT ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────┐         HTTP/WS          ┌──────────────────────┐    │
│   │                 │◄────────────────────────►│                      │    │
│   │    Frontend     │     /next, /label        │      Backend         │    │
│   │   (React/Vite)  │     /points, /ws/logs    │     (FastAPI)        │    │
│   │                 │                          │                      │    │
│   └────────┬────────┘                          └──────────┬───────────┘    │
│            │                                              │                │
│   ┌────────▼────────┐                          ┌──────────▼───────────┐    │
│   │  regl-scatterplot│                         │   PyTorch Model      │    │
│   │  (WebGL 2D viz)  │                         │  (SimpleEmbeddingNet)│    │
│   └─────────────────┘                          └──────────┬───────────┘    │
│                                                           │                │
│                                                ┌──────────▼───────────┐    │
│                                                │      Qdrant          │    │
│                                                │  (Vector Database)   │    │
│                                                └──────────┬───────────┘    │
│                                                           │                │
│                                                ┌──────────▼───────────┐    │
│                                                │     Squeeze UMAP     │    │
│                                                │  (2D Projection)     │    │
│                                                └──────────────────────┘    │
│                                                                             │
│                         ┌──────────────────────┐                           │
│                         │   Modal (Optional)   │                           │
│                         │  (GPU Training Jobs) │                           │
│                         └──────────────────────┘                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

## System Components

### Frontend ([React](https://github.com/facebook/react) + [Vite](https://github.com/vitejs/vite) + TypeScript)

**Location:** `frontend/src/`

The frontend provides an optimized labelling interface designed for speed and minimal cognitive load.

| File | Purpose |
|------|---------|
| `App.tsx` | Main application component with labelling workflow |
| `api.ts` | API client for backend communication |
| `components/ScatterPlot.tsx` | WebGL-powered embedding visualization |
| `components/LogConsole.tsx` | Real-time backend log streaming |
| `components/ui/*` | Shadcn/ui component library |

**Key Features:**

- **Blended Input Interface**: Single input field handles both label confirmation and correction
- **Keyboard-First Design**: `Shift+Enter` for instant confirmation, `J` to accept suggestions
- **History Navigation**: `Alt+←/→` to review and correct previous labels
- **Real-time Visualization**: WebGL scatter plot shows embedding space with color-coded labels
- **WebSocket Logs**: Live streaming of backend training events

**UI Components ([shadcn/ui](https://github.com/shadcn-ui/ui)):**
```
Button, Input, Badge, Card - Modern, accessible components
Tailwind CSS - Zinc-based dark theme
Lucide Icons - Consistent iconography
```

### Backend ([FastAPI](https://github.com/tiangolo/fastapi) + [PyTorch](https://github.com/pytorch/pytorch))

**Location:** `backend/app/`

The backend manages the ML pipeline, vector search, and serves the API.

| File | Purpose |
|------|---------|
| `main.py` | FastAPI application, endpoints, and orchestration |
| `common.py` | Shared models, dataset wrapper, configuration |
| `modal_ops.py` | Optional GPU training via Modal |

**Core Classes:**

```python
# Neural Network (common.py:44)
class SimpleEmbeddingNet(nn.Module):
    """CNN that maps 28x28 grayscale images to 64-dim normalized embeddings"""
    - Conv layers: 1→32→64 channels
    - Output: L2-normalized 64-dimensional vector
    - Optional classification head for supervised learning

# Dataset Wrapper (common.py:78)
class MNISTWrapper:
    """Manages image data, IDs, and user labels"""
    - Loads MNIST subset (configurable, default 1000 images)
    - Assigns UUID to each image
    - Tracks user_labels: Dict[id, label]
    - Provides base64 encoding for frontend display
```

**API Endpoints:**

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves frontend (production) |
| `/next` | GET | Returns next image to label with AI suggestion |
| `/label` | POST | Submits user label, triggers training |
| `/vote` | POST | Submits same/different feedback |
| `/points` | GET | Returns 2D projection for visualization |
| `/ws/logs` | WS | WebSocket for real-time log streaming |

### Vector Database ([Qdrant](https://github.com/qdrant/qdrant))

**Configuration:** Local file-based storage at `./qdrant_data`

[Qdrant](https://github.com/qdrant/qdrant-client) stores 64-dimensional image embeddings and enables fast similarity search.

```python
# Collection setup (main.py:122-127)
qdrant.recreate_collection(
    collection_name="image_embeddings",
    vectors_config=VectorParams(
        size=64,           # EMBEDDING_DIM
        distance=Distance.COSINE
    )
)
```

**Usage:**
- KNN search to find neighbors of current image
- Neighbor label distribution generates suggestions
- Re-indexed when model updates

### Dimensionality Reduction ([Squeeze](https://github.com/GeorgePearse/squeeze))

**Location:** `external/squeeze/`

[Squeeze](https://github.com/GeorgePearse/squeeze) is a high-performance UMAP implementation used to project 64D embeddings to 2D for visualization.

```python
# Projection (main.py:228-229)
reducer = squeeze.UMAP(n_components=2)
embedding_2d = reducer.fit_transform(all_embeddings)
```

### Optional: [Modal](https://github.com/modal-labs/modal-client) GPU Training

**Location:** `backend/app/modal_ops.py`

For faster training, the system can offload GPU computation to [Modal's](https://modal.com/) serverless infrastructure.

```python
@app.function(image=image, gpu="any", timeout=600)
def train_and_reindex(model_state, feedback_buffer, label_buffer, ...):
    # Contrastive loss from same/different votes
    # Classification loss from labels
    # Returns updated model state and re-computed embeddings
```

## The "Chain Reaction" Loop

The core innovation enabling rapid labelling:

```
┌─────────────────────────────────────────────────────────────────┐
│                     CHAIN REACTION WORKFLOW                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. User labels Image A as "Cat"                               │
│                    │                                            │
│                    ▼                                            │
│  2. System queries Qdrant for nearest UNLABELLED               │
│     neighbor of Image A → finds Image B                        │
│                    │                                            │
│                    ▼                                            │
│  3. System checks neighbors of Image B                         │
│     → Image A is close and labeled "Cat"                       │
│     → Suggests "Cat" for Image B                               │
│                    │                                            │
│                    ▼                                            │
│  4. Frontend shows Image B with suggestion                     │
│     "Is it Cat? [J] Accept"                                    │
│                    │                                            │
│                    ▼                                            │
│  5. User presses J or Shift+Enter                              │
│     → Instant confirmation                                      │
│                    │                                            │
│                    ▼                                            │
│  6. Repeat: Find neighbor of B, continue chain                 │
│     until cluster is fully labelled                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Implementation (main.py:301-378):**

1. Check `last_labeled_id` for chain continuation
2. Query Qdrant for 100 nearest neighbors
3. Filter for unlabelled images
4. Select closest unlabelled as next candidate
5. Check candidate's neighbors for label distribution
6. Return most common neighbor label as suggestion

## Data Flow

### Initialization Sequence

```
Application Start
       │
       ├──► Load MNIST dataset (1000 images)
       │    └──► Assign UUIDs to each image
       │
       ├──► Initialize/Load PyTorch model
       │    └──► SimpleEmbeddingNet (64-dim output)
       │
       ├──► Create Qdrant collection
       │    └──► Recreate on each start (UUIDs are ephemeral)
       │
       ├──► Initial index update
       │    └──► Embed all images, store in Qdrant
       │
       └──► Compute initial 2D projection
            └──► Squeeze UMAP: 64D → 2D
```

### Labelling Request Flow

```
GET /next
    │
    ├──► Check last_labeled_id
    │    │
    │    ├──► If exists: Query Qdrant for neighbors
    │    │    └──► Find first unlabelled neighbor (chain)
    │    │
    │    └──► If not: Get random unlabelled image
    │
    ├──► Embed selected image
    │
    ├──► Query neighbors for label distribution
    │
    └──► Return {image, suggestion, debug_info}


POST /label
    │
    ├──► Store label in dataset.user_labels
    │
    ├──► Update last_labeled_id (for chaining)
    │
    ├──► Add to label_buffer
    │
    └──► Check training trigger
         └──► If accumulated_loss >= 0.5: Trigger Modal update
```

### Training Pipeline

```
Training Trigger
       │
       ├──► Snapshot feedback_buffer and label_buffer
       │
       ├──► Clear buffers, reset accumulated_loss
       │
       └──► Async Modal job
            │
            ├──► Contrastive loss (same/different pairs)
            │    loss = target * dist² + (1-target) * relu(margin-dist)²
            │
            ├──► Classification loss (cross-entropy)
            │
            ├──► Backprop and update model
            │
            ├──► Re-embed all images
            │
            └──► Return to main thread
                 │
                 ├──► Update local model state
                 ├──► Save checkpoint
                 └──► Upsert embeddings to Qdrant
```

## Configuration

Key constants defined in `backend/app/common.py`:

| Constant | Value | Description |
|----------|-------|-------------|
| `EMBEDDING_DIM` | 64 | Dimensionality of image embeddings |
| `NUM_IMAGES` | 1000 | Number of MNIST images to load |
| `NUM_CLASSES` | 10 | MNIST digit classes (0-9) |
| `COLLECTION_NAME` | "image_embeddings" | Qdrant collection name |

Runtime configuration in `backend/app/main.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `UPDATE_ON_EVERY_VOTE` | False | Train after every interaction |
| `QDRANT_PATH` | "./qdrant_data" | Vector DB storage location |
| `MODEL_PATH` | "./model_checkpoint.pth" | Model checkpoint file |

## Technology Stack Summary

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | [React 18](https://github.com/facebook/react) + [Vite](https://github.com/vitejs/vite) | Fast development, optimized builds |
| UI Components | [shadcn/ui](https://github.com/shadcn-ui/ui) + [Tailwind](https://github.com/tailwindlabs/tailwindcss) | Modern, accessible design system |
| Visualization | [regl-scatterplot](https://github.com/flekschas/regl-scatterplot) | WebGL 2D scatter plot |
| API | [FastAPI](https://github.com/tiangolo/fastapi) | Async Python web framework |
| ML Framework | [PyTorch](https://github.com/pytorch/pytorch) | Neural network training |
| Vector DB | [Qdrant](https://github.com/qdrant/qdrant) | Similarity search |
| Dim. Reduction | [Squeeze](https://github.com/GeorgePearse/squeeze) (UMAP) | 2D projection |
| GPU Compute | [Modal](https://github.com/modal-labs/modal-client) (optional) | Serverless GPU training |
