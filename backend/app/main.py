import os
import numpy as np
import torch
import torch.optim as optim
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import logging
import ssl
import asyncio
import time
import threading
import uuid
from typing import List
from pydantic import BaseModel
from .simulation import SimulationRunner, STRATEGIES as SIM_STRATEGIES

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
from .oracle import get_strategy

# --- Configuration ---
UPDATE_ON_EVERY_VOTE = False
QDRANT_PATH = "./qdrant_data"
MODEL_PATH = "./model_checkpoint.pth"
SELECTION_STRATEGY = os.environ.get(
    "SELECTION_STRATEGY", "cluster_chain"
)  # cluster_chain, random, uncertainty, margin, diversity

# KNN configuration for suggestions (configurable at runtime)
knn_config = {
    "k_neighbors": int(os.environ.get("KNN_K", "5")),  # Default to 5 for faster suggestions
    "min_k": 1,
    "max_k": 100,
}

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
            except Exception:
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

from collections import Counter  # noqa: E402
import sys  # noqa: E402

# Add external/squeeze to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "../..")
squeeze_path = os.path.join(project_root, "external/squeeze")
if squeeze_path not in sys.path:
    sys.path.append(squeeze_path)

import squeeze  # noqa: E402

# --- State ---
accumulated_loss = 0.0
is_training = False
feedback_buffer = []
label_buffer = []

# Cache for 2D points
cached_points = []
last_labeled_id = None

# Startup status tracking
startup_status = {
    "ready": False,
    "stage": "starting",
    "progress": 0,
    "message": "Starting up..."
}

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
# Always recreate collection since dataset is ephemeral (UUIDs change on restart)
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
    global startup_status
    startup_status["stage"] = "embedding"
    startup_status["message"] = "Generating embeddings..."
    logger.info("Performing initial index update...")

    model.eval()
    batch_size = 128
    points = []
    total_batches = (NUM_IMAGES + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, NUM_IMAGES, batch_size)):
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

            # Update progress
            progress = int((batch_idx + 1) / total_batches * 50)  # 0-50% for embedding
            startup_status["progress"] = progress
            startup_status["message"] = f"Generating embeddings... {end_idx}/{NUM_IMAGES}"
            logger.info(f"Embedding progress: {end_idx}/{NUM_IMAGES} ({progress}%)")

    startup_status["stage"] = "indexing"
    startup_status["message"] = "Indexing vectors..."
    for i in range(0, len(points), 100):
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i : i + 100])
    logger.info("Initial index update complete.")


def compute_predictions_for_points(points_list):
    """
    For each unlabeled point, predict its label based on k nearest labeled neighbors.
    Returns points with added 'predicted_label' and 'confidence' fields.
    """
    if not dataset.user_labels:
        # No labels yet, return points as-is
        return points_list

    # Get all embeddings for KNN
    labeled_ids = set(dataset.user_labels.keys())

    updated_points = []
    for point in points_list:
        pid = point["id"]

        if point.get("label"):
            # Already labeled - full confidence
            updated_points.append({
                **point,
                "predicted_label": point["label"],
                "confidence": 1.0
            })
        else:
            # Unlabeled - predict from neighbors
            idx = dataset.get_idx_by_id(pid)
            if idx is None:
                updated_points.append(point)
                continue

            # Get embedding and query Qdrant
            tensor = dataset.get_tensor(idx).unsqueeze(0)
            with torch.no_grad():
                emb = model(tensor).cpu().squeeze().numpy()

            # Find nearest neighbors
            hits = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=emb,
                limit=20  # Look at more neighbors for better confidence estimate
            ).points

            # Count labeled neighbors
            neighbor_labels = []
            neighbor_scores = []
            for h in hits:
                if h.id == pid:
                    continue
                if h.id in labeled_ids:
                    neighbor_labels.append(dataset.user_labels[h.id])
                    neighbor_scores.append(h.score)  # Cosine similarity

            if neighbor_labels:
                # Compute weighted vote (closer neighbors have more weight)
                label_weights = {}
                for label, score in zip(neighbor_labels, neighbor_scores):
                    label_weights[label] = label_weights.get(label, 0) + score

                # Get prediction and confidence
                predicted_label = max(label_weights, key=label_weights.get)
                total_weight = sum(label_weights.values())
                confidence = label_weights[predicted_label] / total_weight if total_weight > 0 else 0

                # Scale confidence: also consider how many labeled neighbors we found
                # More labeled neighbors = more confident
                coverage = min(len(neighbor_labels) / 5, 1.0)  # Cap at 5 neighbors
                confidence = confidence * coverage

                updated_points.append({
                    **point,
                    "predicted_label": predicted_label,
                    "confidence": round(confidence, 3)
                })
            else:
                # No labeled neighbors nearby
                updated_points.append({
                    **point,
                    "predicted_label": None,
                    "confidence": 0
                })

    return updated_points


def update_projection():
    global cached_points, startup_status
    startup_status["stage"] = "projection"
    startup_status["progress"] = 55
    startup_status["message"] = "Computing 2D projection..."
    logger.info("Updating 2D projection with squeeze...")

    # Get all embeddings
    # We will just run inference on all images to be safe and simple
    model.eval()
    all_embeddings = []

    batch_size = 128
    total_batches = (NUM_IMAGES + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx, i in enumerate(range(0, NUM_IMAGES, batch_size)):
            end_idx = min(i + batch_size, NUM_IMAGES)
            # dataset.get_tensor uses device, so this is on GPU/MPS
            tensors = torch.stack([dataset.get_tensor(j) for j in range(i, end_idx)])
            embeddings = model(tensors).cpu().numpy()
            all_embeddings.append(embeddings)

            # Update progress (55-70% for gathering embeddings)
            progress = 55 + int((batch_idx + 1) / total_batches * 15)
            startup_status["progress"] = progress
            startup_status["message"] = f"Gathering embeddings for projection... {end_idx}/{NUM_IMAGES}"

    all_embeddings = np.vstack(all_embeddings)

    # Run Squeeze UMAP
    startup_status["progress"] = 70
    startup_status["message"] = "Running UMAP dimensionality reduction..."
    logger.info("Running Squeeze UMAP...")

    reducer = squeeze.UMAP(n_components=2)
    embedding_2d = reducer.fit_transform(all_embeddings)

    startup_status["progress"] = 95
    startup_status["message"] = "Formatting points..."

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
    startup_status["progress"] = 100
    startup_status["ready"] = True
    startup_status["stage"] = "ready"
    startup_status["message"] = "Ready!"
    logger.info(f"Projection complete. {len(points)} points.")


# Run initial projection
# We'll run this after model load
# Wait for initial_index_update to complete if running concurrently?
# But here we run them sequentially on startup.

# Initialize selection strategy
selection_strategy = get_strategy(SELECTION_STRATEGY)
logger.info(f"Using selection strategy: {SELECTION_STRATEGY}")

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


@app.get("/status")
def get_status():
    """Return the current startup/initialization status."""
    return startup_status


@app.get("/config/knn")
def get_knn_config():
    """Return current KNN configuration."""
    return knn_config


@app.post("/config/knn")
def set_knn_config(k: int):
    """Set the K value for KNN suggestions."""
    if k < knn_config["min_k"] or k > knn_config["max_k"]:
        return {"error": f"K must be between {knn_config['min_k']} and {knn_config['max_k']}"}
    knn_config["k_neighbors"] = k
    logger.info(f"KNN K updated to {k}")
    return knn_config


@app.get("/config/strategy")
def get_current_strategy():
    """Return current active selection strategy."""
    return {"strategy": SELECTION_STRATEGY, "available": list(SIM_STRATEGIES.keys())}


@app.post("/config/strategy")
def set_current_strategy(strategy: str):
    """Update the active selection strategy."""
    global SELECTION_STRATEGY, selection_strategy
    
    # Handle direct string body or query param
    # If the user sends a raw string in body, FastAPI might expect JSON. 
    # Let's support query param for simplicity or expect a JSON if needed.
    # Actually, for a simple string, query param is easiest.
    
    if strategy not in SIM_STRATEGIES:
         return {"error": f"Invalid strategy. Available: {list(SIM_STRATEGIES.keys())}"}
    
    SELECTION_STRATEGY = strategy
    selection_strategy = get_strategy(strategy)
    logger.info(f"Updated selection strategy to: {strategy}")
    return {"status": "ok", "strategy": strategy}


@app.get("/points")
def get_points(predictions: bool = False):
    """
    Return 2D projection points.

    Args:
        predictions: If True, include predicted labels and confidence for unlabeled points.
                    This is slower but enables confidence-based visualization.
    """
    if not cached_points:
        update_projection()

    # Update labels from current user_labels (cached_points may have stale labels)
    points_with_labels = []
    for p in cached_points:
        points_with_labels.append({
            **p,
            "label": dataset.user_labels.get(p["id"])
        })

    if predictions:
        return compute_predictions_for_points(points_with_labels)

    return points_with_labels


@app.get("/next")
def get_next_sample():
    # Use the configured selection strategy
    candidate_id = selection_strategy.select_next(
        model=model,
        dataset=dataset,
        qdrant_client=qdrant,
        collection_name=COLLECTION_NAME,
        user_labels=dataset.user_labels,
        last_labeled_id=last_labeled_id,
    )

    if candidate_id is None:
        return {"status": "done", "message": "All images labelled!"}

    idx1 = dataset.get_idx_by_id(candidate_id)
    if idx1 is None:
        logger.error(
            f"Candidate ID {candidate_id} not found in dataset! This implies Qdrant/Dataset mismatch."
        )
        return {"status": "error", "message": "Could not find valid image index."}

    # 3. Get embedding
    tensor1 = dataset.get_tensor(idx1).unsqueeze(0)
    with torch.no_grad():
        emb1 = model(tensor1).cpu().squeeze().numpy()

    # 4. Query Qdrant for neighbors (using configurable K)
    hits = qdrant.query_points(
        collection_name=COLLECTION_NAME, query=emb1, limit=knn_config["k_neighbors"]
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
    sorted_suggestions = []
    if neighbor_labels:
        counts = Counter(neighbor_labels)
        suggestion = counts.most_common(1)[0][0]
        # Get all labels sorted by frequency
        sorted_suggestions = [label for label, _ in counts.most_common()]

    return {
        "image": {"id": candidate_id, "data": dataset.get_base64(idx1)},
        "suggestion": suggestion,
        "suggestions": sorted_suggestions,
        "debug_info": {
            "neighbor_count": len(neighbor_labels),
            "neighbors": neighbor_labels,
            "k_neighbors": knn_config["k_neighbors"],
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



# --- Simulation Endpoints ---
sim_results = {}  # {sim_id: {status: running|done|error, results: {...}}}

class SimulationRequest(BaseModel):
    strategy: str
    num_images: int = 100
    max_labels: int = 100

@app.post("/simulation/run")
def run_simulation(request: SimulationRequest):
    """Start a simulation in the background."""
    sim_id = str(uuid.uuid4())
    sim_results[sim_id] = {
        "status": "running", 
        "request": request.dict(),
        "start_time": time.time()
    }

    def _run_sim():
        try:
            logger.info(f"Starting simulation {sim_id} with strategy {request.strategy}")
            runner = SimulationRunner(
                strategy_name=request.strategy,
                num_images=request.num_images,
                use_qdrant_memory=True
            )
            # Run simulation
            metrics = runner.run(
                max_labels=request.max_labels,
                verbose=True,
                log_every=10
            )
            sim_results[sim_id]["status"] = "done"
            sim_results[sim_id]["results"] = metrics
            sim_results[sim_id]["duration"] = time.time() - sim_results[sim_id]["start_time"]
            logger.info(f"Simulation {sim_id} finished successfully")
        except Exception as e:
            logger.error(f"Simulation {sim_id} failed: {e}")
            sim_results[sim_id]["status"] = "error"
            sim_results[sim_id]["error"] = str(e)

    # Run in a separate thread to not block API
    thread = threading.Thread(target=_run_sim)
    thread.daemon = True
    thread.start()

    return {"sim_id": sim_id, "status": "started"}

@app.get("/simulation/results")
def get_simulation_results():
    """Get all simulation results."""
    # Convert dict to list sorted by start time
    results_list = []
    for sid, data in sim_results.items():
        results_list.append({"id": sid, **data})
    
    # Sort by start time descending
    results_list.sort(key=lambda x: x["start_time"], reverse=True)
    return results_list

@app.get("/simulation/strategies")
def get_simulation_strategies():
    """Get available strategies."""
    return list(SIM_STRATEGIES.keys())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
