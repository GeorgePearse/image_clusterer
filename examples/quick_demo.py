"""
Quick Demo: Oracle-based Simulation

This script demonstrates the oracle and selection strategy system
in a simple, self-contained example. Great for understanding the
core concepts before diving into the full simulation framework.

Usage:
    python examples/quick_demo.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.app.common import MNISTWrapper, SimpleEmbeddingNet, EMBEDDING_DIM, device
from backend.app.oracle import Oracle, get_strategy
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import torch
import numpy as np


def run_demo():
    print("=" * 60)
    print("QuickSort Oracle Demo")
    print("=" * 60)

    # Setup
    print("\n1. Loading dataset (100 images)...")
    dataset = MNISTWrapper(num_images=100)

    print("2. Initializing model...")
    model = SimpleEmbeddingNet(EMBEDDING_DIM).to(device)
    model.eval()

    print("3. Setting up Qdrant (in-memory)...")
    qdrant = QdrantClient(":memory:")
    collection_name = "demo"

    qdrant.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    # Index images
    print("4. Computing embeddings...")
    points = []
    with torch.no_grad():
        for i in range(len(dataset.ids)):
            tensor = dataset.get_tensor(i).unsqueeze(0)
            emb = model(tensor).cpu().squeeze().numpy()
            points.append(
                PointStruct(
                    id=dataset.ids[i], vector=emb.tolist(), payload={"index": i}
                )
            )

    qdrant.upsert(collection_name=collection_name, points=points)

    print("5. Creating oracle (perfect labeler)...")
    oracle = Oracle(dataset, noise_rate=0.0)

    # Compare strategies
    print("\n" + "=" * 60)
    print("Comparing Strategies (20 labels each)")
    print("=" * 60)

    strategies_to_test = ["random", "cluster_chain", "uncertainty"]

    for strategy_name in strategies_to_test:
        # Reset dataset labels
        dataset.user_labels = {}

        # Create strategy
        strategy = get_strategy(strategy_name)
        last_labeled_id = None

        print(f"\n{strategy_name.upper()}")
        print("-" * 40)

        # Label 20 images
        for step in range(20):
            # Select next image
            img_id = strategy.select_next(
                model=model,
                dataset=dataset,
                qdrant_client=qdrant,
                collection_name=collection_name,
                user_labels=dataset.user_labels,
                last_labeled_id=last_labeled_id,
            )

            if img_id is None:
                print(f"  All images labeled at step {step}!")
                break

            # Oracle labels it
            label = oracle.label(img_id)
            dataset.user_labels[img_id] = label
            last_labeled_id = img_id

            # Show first few selections
            if step < 5:
                idx = dataset.get_idx_by_id(img_id)
                true_label = dataset.labels[idx]
                print(
                    f"  Step {step + 1}: Selected image {idx}, label={label} (true={true_label})"
                )

        # Calculate label distribution
        from collections import Counter

        label_dist = Counter(dataset.user_labels.values())
        print(f"\n  Labels collected: {len(dataset.user_labels)}")
        print(f"  Distribution: {dict(label_dist)}")

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nKey Observations:")
    print("- Random: Scattered across all classes")
    print("- Cluster Chain: Focuses on nearby images (chains through clusters)")
    print("- Uncertainty: Targets ambiguous examples")
    print("\nTry the full simulation with: python -m backend.simulate")


if __name__ == "__main__":
    run_demo()
