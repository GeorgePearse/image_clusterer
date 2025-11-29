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
from PIL import Image, ImageDraw
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import logging
import ssl
import asyncio
from typing import List

# Fix for SSL certificate verify failed
ssl._create_default_https_context = ssl._create_unverified_context


# --- Log Streaming Setup ---
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
        # We need to run the async broadcast in the existing event loop if possible
        # However, logging is synchronous.
        # We can use a helper to schedule it or just print if loop is not ready.
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                asyncio.run_coroutine_threadsafe(manager.broadcast(log_entry), loop)
        except RuntimeError:
            pass  # Loop might not be running yet


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Add WebSocket handler
ws_handler = WebSocketLogHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
ws_handler.setFormatter(formatter)
logging.getLogger().addHandler(ws_handler)
# Also add to uvicorn logger to capture server events
logging.getLogger("uvicorn").addHandler(ws_handler)


# --- Configuration ---
EMBEDDING_DIM = 64
NUM_IMAGES = 1000
# Dream Mode: Update after every single interaction
UPDATE_ON_EVERY_VOTE = True
COLLECTION_NAME = "image_embeddings"
QDRANT_PATH = "./qdrant_data"
MODEL_PATH = "./model_checkpoint.pth"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Device configuration
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
logger.info(f"Using device: {device}")

# --- State ---
accumulated_loss = 0.0


# --- Model ---
class SimpleEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=64):
        super(SimpleEmbeddingNet, self).__init__()
        # MNIST is 1 channel (grayscale), 28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 28 -> 14 -> 7
        self.fc = nn.Linear(64 * 7 * 7, embedding_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        # Normalize embedding to hypersphere
        return torch.nn.functional.normalize(x, p=2, dim=1)

    def get_embedding(self, x):
        return self.forward(x)


# --- Dataset ---
class MNISTWrapper:
    def __init__(self, num_images=None):
        from torchvision.datasets import MNIST

        # Download MNIST
        self.dataset = MNIST(
            root="./backend/data", train=True, download=True, transform=None
        )

        self.images = []
        self.ids = []
        self.labels = []  # For debugging/logic (not used for training unless cheating)

        # Select subset
        total_available = len(self.dataset)
        if num_images is None:
            num_images = total_available
        else:
            num_images = min(num_images, total_available)

        indices = np.random.choice(total_available, num_images, replace=False)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # MNIST mean/std approx
            ]
        )

        print(f"Loading {num_images} MNIST images...")
        for i in indices:
            img, label = self.dataset[i]
            self.images.append(img)
            self.labels.append(label)
            self.ids.append(str(uuid.uuid4()))

    def get_tensor(self, idx):
        # Return on device
        return self.transform(self.images[idx]).to(device)

    def get_base64(self, idx):
        img = self.images[idx]
        img = img.resize((128, 128), Image.NEAREST)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def get_id(self, idx):
        return self.ids[idx]

    def get_idx_by_id(self, id):
        try:
            return self.ids.index(id)
        except ValueError:
            return None


# --- Application State ---
# Initialize MNIST
try:
    dataset = MNISTWrapper(NUM_IMAGES)
except Exception as e:
    logger.error(f"Failed to load MNIST: {e}")
    # Fallback or exit?
    raise e

model = SimpleEmbeddingNet(EMBEDDING_DIM).to(device)


if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        logger.info("Loaded model checkpoint.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Persistent Qdrant
qdrant = QdrantClient(path=QDRANT_PATH)

# Check if collection exists, if not create
collections = qdrant.get_collections().collections
if not any(c.name == COLLECTION_NAME for c in collections):
    qdrant.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

feedback_buffer = []


def update_embeddings_index():
    logger.info("Updating Qdrant index...")
    model.eval()

    # Process in batches
    batch_size = 128
    points = []

    with torch.no_grad():
        for i in range(0, NUM_IMAGES, batch_size):
            end_idx = min(i + batch_size, NUM_IMAGES)
            # Stack batch
            tensors = torch.stack([dataset.get_tensor(j) for j in range(i, end_idx)])
            embeddings = model(tensors).cpu().numpy()

            for k, emb in enumerate(embeddings):
                idx = i + k
                points.append(
                    PointStruct(
                        id=dataset.ids[idx], vector=emb.tolist(), payload={"index": idx}
                    )
                )

    # Batch upsert to Qdrant
    upsert_batch_size = 100
    for i in range(0, len(points), upsert_batch_size):
        qdrant.upsert(
            collection_name=COLLECTION_NAME, points=points[i : i + upsert_batch_size]
        )
    logger.info("Index update complete.")


# Initialize index on startup
update_embeddings_index()

# --- API ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Resolve absolute path to frontend
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
# Serve built frontend from 'frontend/dist'
frontend_dir = os.path.join(current_dir, "../../frontend/dist")

# We mount root / to serve index.html for SPA support, but FastAPI static files don't do SPA routing natively well.
# Standard pattern: Mount /assets to /assets, and catch-all to serve index.html
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
    return {"message": "Frontend not built. Run 'npm run build' in frontend directory."}


from fastapi.responses import FileResponse


@app.websocket("/ws/logs")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # Keep connection open
    except WebSocketDisconnect:
        manager.disconnect(websocket)


class VoteRequest(BaseModel):
    id1: str
    id2: str
    are_same: bool


@app.get("/pair")
def get_pair():
    # Active Learning Strategy:
    # We want to find pairs that the model is uncertain about or that are hard.
    # Simple hard negative mining: Find pairs that are close in distance but might be different.
    # OR randomly explore.

    strategy = random.choice(["random", "hard", "hard"])  # Bias towards hard examples

    if strategy == "random":
        idx1, idx2 = random.sample(range(NUM_IMAGES), 2)
    else:
        # Pick a random anchor
        idx1 = random.randint(0, NUM_IMAGES - 1)
        tensor1 = dataset.get_tensor(idx1).unsqueeze(0)
        with torch.no_grad():
            emb1 = model(tensor1).cpu().squeeze().numpy()

        # Search for similar items (Hard Positives or Hard Negatives)
        # Since we don't know the true labels here (in the real world), we rely on
        # finding items that are close. If they are close, the user saying "NO" provides a strong signal (Push apart).
        # If they are close and user says "YES", it reinforces.
        hits = qdrant.query_points(
            collection_name=COLLECTION_NAME, query=emb1, limit=5
        ).points

        # Pick one from the top hits (excluding itself)
        candidates = [h for h in hits if h.id != dataset.ids[idx1]]
        if candidates:
            # Pick a random candidate from top k to add some variety
            target = random.choice(candidates)
            id2 = target.id
            idx2 = dataset.get_idx_by_id(id2)
        else:
            # Fallback to random
            idx2 = random.randint(0, NUM_IMAGES - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, NUM_IMAGES - 1)

    return {
        "image1": {"id": dataset.get_id(idx1), "data": dataset.get_base64(idx1)},
        "image2": {"id": dataset.get_id(idx2), "data": dataset.get_base64(idx2)},
        "debug_strategy": strategy,
    }


@app.post("/vote")
def vote(request: VoteRequest):
    global accumulated_loss

    feedback_buffer.append(request)

    # Calculate loss for this item to decide if we should update
    idx1 = dataset.get_idx_by_id(request.id1)
    idx2 = dataset.get_idx_by_id(request.id2)

    if idx1 is not None and idx2 is not None:
        model.eval()
        with torch.no_grad():
            t1 = dataset.get_tensor(idx1).unsqueeze(0)
            t2 = dataset.get_tensor(idx2).unsqueeze(0)
            emb1 = model(t1)
            emb2 = model(t2)

            # Loss calculation (same as training)
            target = 1.0 if request.are_same else 0.0
            margin = 1.0
            dist = torch.nn.functional.pairwise_distance(emb1, emb2)

            # Single item loss
            # Loss = y * D^2 + (1-y) * max(margin - D, 0)^2
            loss_val = target * (dist**2) + (1 - target) * (
                torch.nn.functional.relu(margin - dist) ** 2
            )

            accumulated_loss += loss_val.item()
            logger.info(
                f"Vote received. Item Loss: {loss_val.item():.4f}, Accumulated Loss: {accumulated_loss:.4f}, Buffer: {len(feedback_buffer)}"
            )

    # Trigger update logic
    should_update = False
    if UPDATE_ON_EVERY_VOTE:
        should_update = True
    elif accumulated_loss >= 0.5:  # Hardcoded threshold fallback
        should_update = True

    if should_update:
        logger.info(
            f"Triggering update. Mode: {'Every Vote' if UPDATE_ON_EVERY_VOTE else 'Threshold'}"
        )
        train_step()
        feedback_buffer.clear()
        accumulated_loss = 0.0  # Reset
        # Save model
        torch.save(model.state_dict(), MODEL_PATH)
        # Update index
        update_embeddings_index()

    return {"status": "ok", "buffer_size": len(feedback_buffer)}


def train_step():
    logger.info("Training step started...")
    model.train()
    optimizer.zero_grad()

    embeddings_a = []
    embeddings_b = []
    labels = []

    valid_pairs = 0
    for item in feedback_buffer:
        idx1 = dataset.get_idx_by_id(item.id1)
        idx2 = dataset.get_idx_by_id(item.id2)

        if idx1 is None or idx2 is None:
            continue

        t1 = dataset.get_tensor(idx1)
        t2 = dataset.get_tensor(idx2)

        embeddings_a.append(model(t1.unsqueeze(0)))
        embeddings_b.append(model(t2.unsqueeze(0)))

        # Contrastive Loss: 1 = same, 0 = different
        # We assume 1 means "pull together", 0 means "push apart"
        labels.append(1 if item.are_same else 0)
        valid_pairs += 1

    if valid_pairs == 0:
        return

    embeddings_a = torch.cat(embeddings_a)
    embeddings_b = torch.cat(embeddings_b)
    target = torch.tensor(labels, dtype=torch.float32).to(device)

    # Manual Contrastive Loss
    margin = 1.0
    dist = torch.nn.functional.pairwise_distance(embeddings_a, embeddings_b)

    loss_contrastive = torch.mean(
        (target) * torch.pow(dist, 2)
        + (1 - target) * torch.pow(torch.nn.functional.relu(margin - dist), 2)
    )

    loss_contrastive.backward()
    optimizer.step()
    logger.info(f"Training step complete. Loss: {loss_contrastive.item()}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
