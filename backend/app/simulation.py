"""
Simulation Framework for QuickSort Selection Strategies

This script runs automated simulations to compare different selection strategies
using an Oracle (perfect labeler). Can run locally or on remote compute (e2b/Modal).

Usage:
    # Run all strategies
    python -m backend.simulate

    # Run specific strategy
    python -m backend.simulate --strategy cluster_chain

    # Run with custom parameters
    python -m backend.simulate --num-images 1000 --num-labels 100

    # Save detailed results
    python -m backend.simulate --output results.json
"""

import argparse
import json
import time
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import torch
from collections import Counter, defaultdict
from sklearn.neighbors import KNeighborsClassifier

# Add project root to path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.join(current_dir, "..")
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

from .common import (
    SimpleEmbeddingNet,
    MNISTWrapper,
    EMBEDDING_DIM,
    NUM_CLASSES,
    device,
)
from .oracle import Oracle, get_strategy, STRATEGIES
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter, FieldCondition, MatchValue
from qdrant_client.http.models import QueryRequest


class SimulationRunner:
    """Runs and tracks a single simulation."""

    def __init__(
        self,
        strategy_name: str,
        num_images: int = 1000,
        embedding_dim: int = EMBEDDING_DIM,
        use_qdrant_memory: bool = True,
    ):
        self.strategy_name = strategy_name
        self.num_images = num_images
        self.embedding_dim = embedding_dim
        self.use_qdrant_memory = use_qdrant_memory

        # Initialize components
        print(f"\n{'=' * 60}")
        print(f"Initializing simulation: {strategy_name}")
        print(f"{'=' * 60}")

        # Dataset
        print(f"Loading {num_images} images...")
        self.dataset = MNISTWrapper(num_images)

        # Model
        print("Initializing model...")
        self.model = SimpleEmbeddingNet(embedding_dim, num_classes=NUM_CLASSES).to(
            device
        )
        self.model.eval()

        # Qdrant
        collection_name = f"sim_{strategy_name}_{int(time.time())}"
        self.collection_name = collection_name

        if use_qdrant_memory:
            print("Initializing Qdrant (in-memory)...")
            self.qdrant = QdrantClient(":memory:")
        else:
            print("Initializing Qdrant (persistent)...")
            self.qdrant = QdrantClient(path="./qdrant_sim_data")

        self.qdrant.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE),
        )

        # Index all images
        print("Computing embeddings and indexing...")
        self.all_embeddings = self._index_all_images()

        # Oracle and Strategy
        print("Initializing oracle and strategy...")
        self.oracle = Oracle(self.dataset, noise_rate=0.0)
        self.strategy = get_strategy(strategy_name)

        # Metrics
        self.metrics = {
            "strategy": strategy_name,
            "num_images": num_images,
            "labels_history": [],  # [(step, num_labeled, accuracy, cluster_purity, cumulative_time)]
            "selection_times": [],
            "labeling_times": [],
            "total_time": 0.0,
            "simulated_human_time": 0.0,
        }

        print("Initialization complete!\n")

    def _get_human_labeling_cost(self, current_label: str, previous_label: str | None) -> float:
        """
        Simulate human labeling time (seconds).
        - Flow State (Same Class): 0.6s (Fast confirmation)
        - Context Switch (Diff Class): 2.5s (Cognitive load + typing)
        """
        if previous_label is None:
            return 2.5 # First label is always "slow"
        
        if current_label == previous_label:
            return 0.6
        return 2.5

    def _index_all_images(self):
        """Compute embeddings and index in Qdrant. Returns embedding matrix."""
        batch_size = 128
        points = []
        all_embeddings = []

        with torch.no_grad():
            for i in range(0, self.num_images, batch_size):
                end_idx = min(i + batch_size, self.num_images)
                tensors = torch.stack(
                    [self.dataset.get_tensor(j) for j in range(i, end_idx)]
                )
                embeddings = self.model(tensors).cpu().numpy()
                all_embeddings.append(embeddings)

                for k, emb in enumerate(embeddings):
                    idx = i + k
                    points.append(
                        PointStruct(
                            id=self.dataset.ids[idx],
                            vector=emb.tolist(),
                            payload={"index": idx, "is_labeled": False},
                        )
                    )

        # Batch upsert
        for i in range(0, len(points), 100):
            self.qdrant.upsert(
                collection_name=self.collection_name, points=points[i : i + 100]
            )
            
        return np.vstack(all_embeddings)

    def _calculate_knn_accuracy(self, k: int = 5) -> float:
        """
        Evaluate KNN classifier on all data using Qdrant batch search.
        Returns accuracy (0.0 - 1.0).
        """
        if len(self.dataset.user_labels) < k:
            return 0.0

        # Construct batch queries for all images
        requests = []
        # Filter: only consider neighbors that have been labeled
        search_filter = Filter(
            must=[
                FieldCondition(key="is_labeled", match=MatchValue(value=True))
            ]
        )

        for vector in self.all_embeddings:
            requests.append(
                QueryRequest(
                    query=vector.tolist(),
                    filter=search_filter,
                    limit=k,
                    with_payload=True
                )
            )
        
        # Execute batch search
        batch_results = self.qdrant.query_batch_points(
            collection_name=self.collection_name,
            requests=requests
        )

        # Calculate accuracy
        correct = 0
        total = len(self.dataset.labels) # Ground truth length

        for i, query_response in enumerate(batch_results):
            # query_response is a QueryResponse object with a 'points' attribute
            if not query_response.points:
                continue
            
            votes = []
            for hit in query_response.points:
                if hit.payload:
                    label = hit.payload.get("label")
                    if label is not None:
                        votes.append(label)
            
            if not votes:
                continue
                
            predicted_label = Counter(votes).most_common(1)[0][0]
            true_label = str(self.dataset.labels[i])
            
            if predicted_label == true_label:
                correct += 1
                
        return correct / total if total > 0 else 0.0

    def _calculate_metrics(self) -> Dict:
        """Calculate current state metrics."""
        if not self.dataset.user_labels:
            return {
                "num_labeled": 0,
                "accuracy": 0.0,
                "knn_accuracy": 0.0,
                "cluster_purity": 0.0,
                "label_distribution": {},
            }

        # Accuracy: % of user labels that match ground truth
        correct = 0
        total = len(self.dataset.user_labels)

        for img_id, user_label in self.dataset.user_labels.items():
            idx = self.dataset.get_idx_by_id(img_id)
            true_label = str(self.dataset.labels[idx])
            if user_label == true_label:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0
        
        # KNN Model Accuracy (K=5)
        knn_accuracy = self._calculate_knn_accuracy(k=5)

        # Cluster purity: For each true class, what % of its labeled examples are correct?
        class_correct = defaultdict(int)
        class_total = defaultdict(int)

        for img_id, user_label in self.dataset.user_labels.items():
            idx = self.dataset.get_idx_by_id(img_id)
            true_label = str(self.dataset.labels[idx])
            class_total[true_label] += 1
            if user_label == true_label:
                class_correct[true_label] += 1

        purities = [
            class_correct[c] / class_total[c] for c in class_total if class_total[c] > 0
        ]
        cluster_purity = np.mean(purities) if purities else 0.0

        # Label distribution
        label_dist = Counter(self.dataset.user_labels.values())

        return {
            "num_labeled": total,
            "accuracy": accuracy,
            "knn_accuracy": knn_accuracy,
            "cluster_purity": cluster_purity,
            "label_distribution": dict(label_dist),
        }

    def run(
        self,
        max_labels: int = None,
        log_every: int = 10,
        verbose: bool = True,
    ) -> Dict:
        """
        Run the simulation.

        Args:
            max_labels: Maximum number of labels to collect (None = all images)
            log_every: Log metrics every N labels
            verbose: Print progress

        Returns:
            Final metrics dictionary
        """
        if max_labels is None:
            max_labels = self.num_images

        start_time = time.time()
        last_labeled_id = None

        for step in range(max_labels):
            # Select next image
            selection_start = time.time()
            img_id = self.strategy.select_next(
                model=self.model,
                dataset=self.dataset,
                qdrant_client=self.qdrant,
                collection_name=self.collection_name,
                user_labels=self.dataset.user_labels,
                last_labeled_id=last_labeled_id,
            )
            selection_time = time.time() - selection_start
            self.metrics["selection_times"].append(selection_time)

            if img_id is None:
                if verbose:
                    print(f"\nAll images labeled at step {step}!")
                break

            # Oracle labels it
            labeling_start = time.time()
            label = self.oracle.label(img_id)
            self.dataset.user_labels[img_id] = label
            
            # Update Payload for filtering
            self.qdrant.set_payload(
                collection_name=self.collection_name,
                payload={"label": label, "is_labeled": True},
                points=[img_id]
            )
            
            last_labeled_id = img_id
            labeling_time = time.time() - labeling_start
            self.metrics["labeling_times"].append(labeling_time)

            # Log metrics
            if (step + 1) % log_every == 0 or step == 0:
                current_metrics = self._calculate_metrics()
                self.metrics["labels_history"].append(
                    {"step": step + 1, **current_metrics}
                )

                if verbose:
                    print(
                        f"Step {step + 1:4d} | "
                        f"Labeled: {current_metrics['num_labeled']:4d} | "
                        f"Accuracy: {current_metrics['accuracy']:.3f} | "
                        f"Purity: {current_metrics['cluster_purity']:.3f} | "
                        f"Select: {selection_time * 1000:.1f}ms"
                    )

        # Final metrics
        self.metrics["total_time"] = time.time() - start_time
        final_state = self._calculate_metrics()
        self.metrics["final_state"] = final_state
        self.metrics["oracle_stats"] = self.oracle.get_stats()

        # Summary stats
        self.metrics["avg_selection_time"] = np.mean(self.metrics["selection_times"])
        self.metrics["avg_labeling_time"] = np.mean(self.metrics["labeling_times"])

        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Simulation Complete: {self.strategy_name}")
            print(f"{'=' * 60}")
            print(f"Total time: {self.metrics['total_time']:.2f}s")
            print(f"Labels collected: {final_state['num_labeled']}")
            print(f"Final accuracy: {final_state['accuracy']:.3f}")
            print(f"Final purity: {final_state['cluster_purity']:.3f}")
            print(
                f"Avg selection time: {self.metrics['avg_selection_time'] * 1000:.1f}ms"
            )

        return self.metrics


def compare_strategies(
    strategies: List[str],
    num_images: int = 1000,
    max_labels: int = None,
    log_every: int = 10,
) -> Dict[str, Dict]:
    """
    Run simulations for multiple strategies and compare.

    Returns:
        Dictionary mapping strategy name to metrics
    """
    results = {}

    for strategy_name in strategies:
        sim = SimulationRunner(
            strategy_name=strategy_name,
            num_images=num_images,
        )
        metrics = sim.run(max_labels=max_labels, log_every=log_every)
        results[strategy_name] = metrics

    # Print comparison
    print(f"\n{'=' * 60}")
    print("STRATEGY COMPARISON")
    print(f"{'=' * 60}")
    print(f"{'Strategy':<20} {'Labels':<10} {'Accuracy':<12} {'Time (s)':<12}")
    print(f"{'-' * 60}")

    for strategy_name, metrics in results.items():
        final = metrics["final_state"]
        print(
            f"{strategy_name:<20} "
            f"{final['num_labeled']:<10} "
            f"{final['accuracy']:<12.3f} "
            f"{metrics['total_time']:<12.2f}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Simulate and compare QuickSort selection strategies"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=list(STRATEGIES.keys()) + ["all"],
        default="all",
        help="Strategy to simulate (default: all)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=1000,
        help="Number of images to use (default: 1000)",
    )
    parser.add_argument(
        "--max-labels",
        type=int,
        default=None,
        help="Maximum labels to collect (default: all images)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log frequency (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    # Determine which strategies to run
    if args.strategy == "all":
        strategies = list(STRATEGIES.keys())
    else:
        strategies = [args.strategy]

    # Run comparison
    results = compare_strategies(
        strategies=strategies,
        num_images=args.num_images,
        max_labels=args.max_labels,
        log_every=args.log_every,
    )

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
