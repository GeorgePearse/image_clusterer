import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import numpy as np
import uuid
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# --- Configuration ---
EMBEDDING_DIM = 64
NUM_IMAGES = 1000
COLLECTION_NAME = "image_embeddings"
NUM_CLASSES = 10  # MNIST


# --- Data Models ---
class VoteRequest(BaseModel):
    id1: str
    id2: str
    are_same: bool


class LabelRequest(BaseModel):
    image_id: str
    label: str  # We might map this to int later, or keep as string and map dynamically


# Device configuration
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# --- Model ---
class SimpleEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=64, num_classes=10):
        super(SimpleEmbeddingNet, self).__init__()
        # MNIST is 1 channel (grayscale), 28x28
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 28 -> 14 -> 7
        self.fc = nn.Linear(64 * 7 * 7, embedding_dim)

        # Classification Head (optional usage)
        # We take the embedding_dim as input.
        # Note: If we use normalized embeddings, the classifier sees points on hypersphere.
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        # Normalize embedding to hypersphere
        return torch.nn.functional.normalize(x, p=2, dim=1)

    def get_embedding(self, x):
        return self.forward(x)

    def forward_with_logits(self, x):
        emb = self.forward(x)
        logits = self.classifier(emb)
        return emb, logits


# --- Dataset ---
class MNISTWrapper:
    def __init__(self, num_images=None, root="./backend/data", download=True):
        from torchvision.datasets import MNIST

        # Download MNIST
        self.dataset = MNIST(root=root, train=True, download=download, transform=None)

        self.images = []
        self.ids = []
        self.labels = []  # Ground truth labels
        self.user_labels = {}  # id -> label

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # MNIST mean/std approx
            ]
        )

        # Select subset
        total_available = len(self.dataset)
        if num_images is None:
            num_images = total_available
        else:
            num_images = min(num_images, total_available)

        indices = np.random.choice(total_available, num_images, replace=False)

        print(f"Loading {num_images} MNIST images...")
        for i in indices:
            img, label = self.dataset[i]
            self.images.append(img)
            self.labels.append(label)
            self.ids.append(str(uuid.uuid4()))

        # Cache tensors on CPU for efficient Modal transfer
        self.cpu_tensors = torch.stack([self.transform(img) for img in self.images])

    def get_tensor(self, idx):
        # Return on device
        return self.cpu_tensors[idx].to(device)

    def get_all_tensors_cpu(self):
        # Return CPU tensors (cached)
        return self.cpu_tensors

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

    def get_unlabelled_id(self):
        # Return a random id that is not in user_labels
        # This is not efficient for large datasets but fine for <10k
        labelled_ids = set(self.user_labels.keys())
        all_ids = set(self.ids)
        unlabelled = list(all_ids - labelled_ids)
        if not unlabelled:
            return None
        return np.random.choice(unlabelled)
