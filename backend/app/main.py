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
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import logging

# --- Configuration ---
EMBEDDING_DIM = 64
NUM_IMAGES = 200  # Increased for better variety
NUM_CLASSES = 10  # Increased classes
COLLECTION_NAME = "image_embeddings"
BATCH_SIZE_FOR_UPDATE = 5
QDRANT_PATH = "./qdrant_data"
MODEL_PATH = "./model_checkpoint.pth"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Model ---
class SimpleEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=64):
        super(SimpleEmbeddingNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, embedding_dim)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.fc(x)
        # Normalize embedding to hypersphere
        return torch.nn.functional.normalize(x, p=2, dim=1)

    def get_embedding(self, x):
        return self.forward(x)


# --- Dataset ---
class SyntheticDataset:
    def __init__(self, num_images, num_classes):
        self.images = []
        self.ids = []
        # We won't store explicit labels to simulate "unlabeled" nature,
        # though we generate them with structure.

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        # Generate images
        for i in range(num_images):
            label = i % num_classes

            # Base color based on label
            hue = label / num_classes
            # Convert HSV to RGB (simplified)
            import colorsys

            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.8)
            base_color = (int(r * 255), int(g * 255), int(b * 255))

            # Create image
            img = Image.new("RGB", (32, 32), color=base_color)
            draw = ImageDraw.Draw(img)

            # Add random shape based on label parity to make it non-trivial
            # Even labels get circles, Odd labels get rectangles
            shape_color = (
                255 - base_color[0],
                255 - base_color[1],
                255 - base_color[2],
            )

            if label % 2 == 0:
                draw.ellipse([8, 8, 24, 24], fill=shape_color)
            else:
                draw.rectangle([8, 8, 24, 24], fill=shape_color)

            # Add noise
            img_arr = np.array(img)
            noise = np.random.randint(-20, 20, img_arr.shape)
            img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
            img = Image.fromarray(img_arr)

            self.images.append(img)
            self.ids.append(str(uuid.uuid4()))

    def get_tensor(self, idx):
        return self.transform(self.images[idx])

    def get_base64(self, idx):
        img = self.images[idx]
        img = img.resize((128, 128), Image.NEAREST)  # Upscale for UI
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
dataset = SyntheticDataset(NUM_IMAGES, NUM_CLASSES)
model = SimpleEmbeddingNet(EMBEDDING_DIM)

if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
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
    points = []
    model.eval()
    with torch.no_grad():
        for i in range(NUM_IMAGES):
            tensor = dataset.get_tensor(i).unsqueeze(0)
            embedding = model(tensor).squeeze().numpy()
            points.append(
                PointStruct(
                    id=dataset.ids[i], vector=embedding.tolist(), payload={"index": i}
                )
            )

    # Batch upsert
    batch_size = 50
    for i in range(0, len(points), batch_size):
        qdrant.upsert(
            collection_name=COLLECTION_NAME, points=points[i : i + batch_size]
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
frontend_dir = os.path.join(current_dir, "../../frontend")

app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")
# Redirect root to /app/index.html
from fastapi.responses import RedirectResponse


@app.get("/")
async def read_root():
    return RedirectResponse(url="/app/index.html")


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
            emb1 = model(tensor1).squeeze().numpy()

        # Search for similar items (Hard Positives or Hard Negatives)
        # Since we don't know the true labels here (in the real world), we rely on
        # finding items that are close. If they are close, the user saying "NO" provides a strong signal (Push apart).
        # If they are close and user says "YES", it reinforces.
        hits = qdrant.search(
            collection_name=COLLECTION_NAME, query_vector=emb1, limit=5
        )

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
    feedback_buffer.append(request)
    logger.info(f"Vote received. Buffer size: {len(feedback_buffer)}")

    if len(feedback_buffer) >= BATCH_SIZE_FOR_UPDATE:
        train_step()
        feedback_buffer.clear()
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
    target = torch.tensor(labels, dtype=torch.float32)

    # Manual Contrastive Loss
    # D = Euclidean Distance
    # Loss = y * D^2 + (1-y) * max(margin - D, 0)^2
    # But we are using Cosine Similarity in Qdrant?
    # Usually consistent to use same metric.
    # If using Cosine, we should minimize (1 - cos) for positives, maximize (1 - cos) for negatives.
    # Let's stick to Euclidean on Normalized Embeddings which is related to Cosine.
    # ||u - v||^2 = 2 - 2(u . v) for normalized vectors.
    # So minimizing Euclidean distance maximizes Cosine Similarity.

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
