import modal
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import sys

# Define the Modal App
app = modal.App("image-clusterer")

# Define the image with dependencies
image = modal.Image.debian_slim().pip_install("torch", "torchvision", "numpy", "Pillow")

# Mount the local backend package so it's available remotely
if hasattr(modal, "Mount"):
    mounts = [modal.Mount.from_local_dir("backend", remote_path="/root/backend")]
else:
    mounts = []

# Modal API change: mounts should be passed to App, or Image, or Function?
# Recent modal versions pass mounts to function.
# If this fails, we will temporarily disable modal ops to fix the UI test first.


@app.function(image=image, gpu="any", timeout=600)
def train_and_reindex(
    model_state, feedback_buffer, label_buffer, all_images_tensor, image_ids
):
    # Setup path for imports inside the container

    if "/root" not in sys.path:
        sys.path.append("/root")

    from backend.app.common import SimpleEmbeddingNet, EMBEDDING_DIM, NUM_CLASSES

    print("Starting train and reindex on Modal...")

    # Reconstruct device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SimpleEmbeddingNet(EMBEDDING_DIM, num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(model_state, strict=False)

    # Training
    if feedback_buffer or label_buffer:
        print(
            f"Training on {len(feedback_buffer)} votes and {len(label_buffer)} labels..."
        )
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Loss functions
        criterion_cls = nn.CrossEntropyLoss()

        # Move all images to device once
        all_images_gpu = all_images_tensor.to(device)

        # Optimization loop
        # Simple approach: one batch update for all data
        optimizer.zero_grad()
        total_loss = 0.0

        # --- Contrastive Loss (Votes) ---
        if feedback_buffer:
            embeddings_a = []
            embeddings_b = []
            labels = []
            valid_pairs = 0

            for item in feedback_buffer:
                try:
                    idx1 = image_ids.index(item.id1)
                    idx2 = image_ids.index(item.id2)
                    t1 = all_images_gpu[idx1]
                    t2 = all_images_gpu[idx2]

                    embeddings_a.append(model(t1.unsqueeze(0)))
                    embeddings_b.append(model(t2.unsqueeze(0)))
                    labels.append(1 if item.are_same else 0)
                    valid_pairs += 1
                except ValueError:
                    continue

            if valid_pairs > 0:
                embeddings_a = torch.cat(embeddings_a)
                embeddings_b = torch.cat(embeddings_b)
                target = torch.tensor(labels, dtype=torch.float32).to(device)

                margin = 1.0
                dist = F.pairwise_distance(embeddings_a, embeddings_b)

                loss_contrastive = torch.mean(
                    (target) * torch.pow(dist, 2)
                    + (1 - target) * torch.pow(F.relu(margin - dist), 2)
                )
                print(f"Contrastive Loss: {loss_contrastive.item()}")
                total_loss += loss_contrastive

        # --- Classification Loss (Labels) ---
        if label_buffer:
            cls_images = []
            cls_targets = []
            valid_labels = 0

            for item in label_buffer:
                try:
                    idx = image_ids.index(item.image_id)
                    # Simple parsing: assume label is digit '0'-'9' for MNIST
                    # If not, we map using hash or fixed map.
                    # Given it's MNIST, let's assume valid int strings.
                    label_val = int(item.label)
                    if 0 <= label_val < NUM_CLASSES:
                        cls_images.append(all_images_gpu[idx].unsqueeze(0))
                        cls_targets.append(label_val)
                        valid_labels += 1
                except (ValueError, IndexError):
                    continue

            if valid_labels > 0:
                cls_images = torch.cat(cls_images)
                cls_targets = torch.tensor(cls_targets, dtype=torch.long).to(device)

                # Forward pass through classifier
                _, logits = model.forward_with_logits(cls_images)

                loss_cls = criterion_cls(logits, cls_targets)
                print(f"Classification Loss: {loss_cls.item()}")
                total_loss += loss_cls

        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()
            optimizer.step()
            print(f"Total Loss: {total_loss.item()}")
        else:
            print("No valid training data.")

    # Re-indexing
    print("Re-indexing all images...")
    model.eval()
    with torch.no_grad():
        if "all_images_gpu" not in locals():
            all_images_gpu = all_images_tensor.to(device)

        # Process in batches
        embeddings = []
        batch_size = 128
        for i in range(0, len(all_images_gpu), batch_size):
            batch = all_images_gpu[i : i + batch_size]
            emb = model(batch)
            embeddings.append(emb.cpu())

        embeddings = torch.cat(embeddings).numpy()

    print("Job complete.")
    return model.state_dict(), embeddings
