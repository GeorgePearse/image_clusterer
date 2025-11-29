import uuid
import io
import base64
import random
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import logging
import ssl
import asyncio
from typing import List

# Import from common and modal_ops
from .common import (
    SimpleEmbeddingNet,
    MNISTWrapper,
    EMBEDDING_DIM,
    NUM_IMAGES,
    COLLECTION_NAME,
    VoteRequest,
    LabelRequest,
    device,
    logger,
)
from .modal_ops import train_and_reindex

# --- Configuration ---
UPDATE_ON_EVERY_VOTE = False
QDRANT_PATH = "./qdrant_data"
MODEL_PATH = "./model_checkpoint.pth"

# Fix SSL
ssl._create_default_https_context = ssl._create_unverified_context


# --- WebSocket Manager ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except:
                pass


manager = ConnectionManager()


class WebSocketLogHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(manager.broadcast(log_entry), loop)
        except RuntimeError:
            pass


logging.basicConfig(level=logging.INFO)
ws_handler = WebSocketLogHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ws_handler.setFormatter(formatter)
logging.getLogger().addHandler(ws_handler)
logging.getLogger("uvicorn").addHandler(ws_handler)

from collections import Counter

import sys
import os

# Add external/squeeze to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../..")
squeeze_path = os.path.join(project_root, "external/squeeze")
if squeeze_path not in sys.path:
    sys.path.append(squeeze_path)

import squeeze

# --- State ---
accumulated_loss = 0.0
is_training = False
feedback_buffer = []
label_buffer = []

# Cache for 2D points
cached_points = []
last_labeled_id = None

# --- Initialization ---
try:
    dataset = MNISTWrapper(NUM_IMAGES)
except Exception as e:
    logger.error(f"Failed to load MNIST: {e}")
    raise e

model = SimpleEmbeddingNet(EMBEDDING_DIM).to(device)
if os.path.exists(MODEL_PATH):
    try:
        # strict=False to handle missing classifier weights if loading old model
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
        logger.info("Loaded model checkpoint.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

# We only need optimizer if we were training locally, but we are not.
# Keeping it in case we want to revert or debug locally, but not strictly needed.
optimizer = optim.Adam(model.parameters(), lr=0.001)

qdrant = QdrantClient(path=QDRANT_PATH)
collections = qdrant.get_collections().collections
if not any(c.name == COLLECTION_NAME for c in collections):
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

# --- Helper Functions ---


async def run_modal_update(vote_buffer_snapshot, label_buffer_snapshot):
    global is_training, accumulated_loss
    is_training = True
    logger.info(
        f"Starting async Modal update with {len(vote_buffer_snapshot)} votes and {len(label_buffer_snapshot)} labels..."
    )

    try:
        # Prepare data for Modal (Fast, cached)
        all_images = dataset.get_all_tensors_cpu()
        ids = dataset.ids

        # Call Modal
        # Using .remote.aio() for async execution
        new_state, new_embeddings = await train_and_reindex.remote.aio(
            model_state=model.state_dict(),
            feedback_buffer=vote_buffer_snapshot,
            label_buffer=label_buffer_snapshot,
            all_images_tensor=all_images,
            image_ids=ids,
        )

        logger.info("Modal update finished. Applying changes locally...")

        # Update local model
        model.load_state_dict(new_state)
        torch.save(new_state, MODEL_PATH)

        # Update Qdrant
        points = []
        for i, emb in enumerate(new_embeddings):
            points.append(
                PointStruct(
                    id=dataset.ids[i], vector=emb.tolist(), payload={"index": i}
                )
            )

        upsert_batch_size = 100
        for i in range(0, len(points), upsert_batch_size):
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + upsert_batch_size],
            )

        logger.info("Local state updated successfully.")

    except Exception as e:
        logger.error(f"Modal update failed: {e}")
    finally:
        is_training = False


def initial_index_update():
    logger.info("Performing initial index update...")
    model.eval()
    batch_size = 128
    points = []
    with torch.no_grad():
        for i in range(0, NUM_IMAGES, batch_size):
            end_idx = min(i + batch_size, NUM_IMAGES)
            tensors = torch.stack([dataset.get_tensor(j) for j in range(i, end_idx)])
            embeddings = model(tensors).cpu().numpy()
            for k, emb in enumerate(embeddings):
                idx = i + k
                points.append(
                    PointStruct(
                        id=dataset.ids[idx], vector=emb.tolist(), payload={"index": idx}
                    )
                )

    for i in range(0, len(points), 100):
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i : i + 100])
    logger.info("Initial index update complete.")


def update_projection():
    global cached_points
    logger.info("Updating 2D projection with squeeze...")

    # Get all embeddings
    # We will just run inference on all images to be safe and simple
    model.eval()
    all_embeddings = []

    batch_size = 128
    with torch.no_grad():
        for i in range(0, NUM_IMAGES, batch_size):
            end_idx = min(i + batch_size, NUM_IMAGES)
            # dataset.get_tensor uses device, so this is on GPU/MPS
            tensors = torch.stack([dataset.get_tensor(j) for j in range(i, end_idx)])
            embeddings = model(tensors).cpu().numpy()
            all_embeddings.append(embeddings)

    all_embeddings = np.vstack(all_embeddings)

    # Run Squeeze UMAP
    reducer = squeeze.UMAP(n_components=2)
    embedding_2d = reducer.fit_transform(all_embeddings)

    # Format points
    points = []
    for i, (x, y) in enumerate(embedding_2d):
        pid = dataset.ids[i]
        points.append(
            {
                "id": pid,
                "x": float(x),
                "y": float(y),
                "label": dataset.user_labels.get(pid),
            }
        )

    cached_points = points
    logger.info(f"Projection complete. {len(points)} points.")


# Run initial projection
# We'll run this after model load
# Wait for initial_index_update to complete if running concurrently?
# But here we run them sequentially on startup.

# Run initial update on startup
initial_index_update()
update_projection()

# --- API ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, "../../frontend/dist")
if os.path.exists(frontend_dir):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(frontend_dir, "assets")),
        name="assets",
    )


@app.get("/")
async def read_root():
    if os.path.exists(os.path.join(frontend_dir, "index.html")):
        return FileResponse(os.path.join(frontend_dir, "index.html"))
    return {"message": "Frontend not built."}


@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@app.get("/points")
def get_points():
    if not cached_points:
        update_projection()
    return cached_points


@app.get("/next")
def get_next_sample():
    # 1. Try to find an unlabelled neighbor of the last labeled point (Cluster Exploration)
    candidate_id = None

    if last_labeled_id:
        idx_last = dataset.get_idx_by_id(last_labeled_id)
        if idx_last is not None:
            # Get embedding of last labeled
            t_last = dataset.get_tensor(idx_last).unsqueeze(0)
            with torch.no_grad():
                emb_last = model(t_last).cpu().squeeze().numpy()

            # Find neighbors
            # Limit 100 to find diverse neighbors
            hits = qdrant.query_points(
                collection_name=COLLECTION_NAME, query=emb_last, limit=100
            ).points

            # Filter for unlabelled
            unlabelled_hits = [h for h in hits if h.id not in dataset.user_labels]

            if unlabelled_hits:
                # Pick the closest one (first one) or a random close one to avoid getting stuck?
                # Picking the closest one is best for "chaining"
                candidate_id = unlabelled_hits[0].id

    # 2. If no candidate (or no history), get random unlabelled
    if candidate_id is None:
        candidate_id = dataset.get_unlabelled_id()

    if candidate_id is None:
        return {"status": "done", "message": "All images labelled!"}

    idx1 = dataset.get_idx_by_id(candidate_id)
    if idx1 is None:
        logger.error(
            f"Candidate ID {candidate_id} not found in dataset! This implies Qdrant/Dataset mismatch."
        )
        # Fallback to random
        candidate_id = dataset.get_unlabelled_id()
        if candidate_id:
            idx1 = dataset.get_idx_by_id(candidate_id)

    if idx1 is None:
        return {"status": "error", "message": "Could not find valid image index."}

    # 3. Get embedding
    tensor1 = dataset.get_tensor(idx1).unsqueeze(0)
    with torch.no_grad():
        emb1 = model(tensor1).cpu().squeeze().numpy()

    # 4. Query Qdrant for neighbors (limit 100 for better context)
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=emb1, limit=100
    ).points

    # 5. Check neighbors for labels
    neighbor_labels = []
    for h in hits:
        if h.id == candidate_id:
            continue
        if h.id in dataset.user_labels:
            neighbor_labels.append(dataset.user_labels[h.id])

    # 6. Suggestion
    suggestion = None
    if neighbor_labels:
        suggestion = Counter(neighbor_labels).most_common(1)[0][0]

    return {
        "image": {"id": candidate_id, "data": dataset.get_base64(idx1)},
        "suggestion": suggestion,
        "debug_info": {
            "neighbor_count": len(neighbor_labels),
            "neighbors": neighbor_labels,
        },
    }


def check_trigger_update():
    global accumulated_loss, is_training
    should_update = False
    if not is_training:
        if UPDATE_ON_EVERY_VOTE:
            should_update = True
        elif accumulated_loss >= 0.5:
            should_update = True

    if should_update:
        # Snapshot buffers
        vote_snapshot = list(feedback_buffer)
        feedback_buffer.clear()

        label_snapshot = list(label_buffer)
        label_buffer.clear()

        accumulated_loss = 0.0

        # Trigger background task
        asyncio.create_task(run_modal_update(vote_snapshot, label_snapshot))


@app.post("/vote")
async def vote(request: VoteRequest):
    global accumulated_loss, is_training

    feedback_buffer.append(request)

    # Calculate loss for logging
    idx1 = dataset.get_idx_by_id(request.id1)
    idx2 = dataset.get_idx_by_id(request.id2)

    if idx1 is not None and idx2 is not None:
        model.eval()
        with torch.no_grad():
            t1 = dataset.get_tensor(idx1).unsqueeze(0)
            t2 = dataset.get_tensor(idx2).unsqueeze(0)
            emb1 = model(t1)
            emb2 = model(t2)
            target = 1.0 if request.are_same else 0.0
            dist = torch.nn.functional.pairwise_distance(emb1, emb2)
            loss_val = target * (dist**2) + (1 - target) * (
                torch.nn.functional.relu(1.0 - dist) ** 2
            )
            accumulated_loss += loss_val.item()
            logger.info(
                f"Vote received. Loss: {loss_val.item():.4f}, Buffer: {len(feedback_buffer)}"
            )

    check_trigger_update()

    return {
        "status": "ok",
        "buffer_size": len(feedback_buffer),
        "training": is_training,
    }


@app.post("/label")
async def label(request: LabelRequest):
    global accumulated_loss, is_training, last_labeled_id

    # Store label locally
    dataset.user_labels[request.image_id] = request.label
    label_buffer.append(request)
    last_labeled_id = request.image_id

    logger.info(f"Label received: {request.label} for {request.image_id}")

    # Trigger update immediately (treat as a vote)
    check_trigger_update()

    return {"status": "ok", "buffer_size": len(label_buffer), "training": is_training}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
