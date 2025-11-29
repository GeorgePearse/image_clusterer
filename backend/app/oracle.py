"""
Oracle and Selection Strategy System

This module provides:
1. Oracle: A simulated perfect labeler that knows ground truth
2. Selection Strategies: Different approaches to choosing the next image to label
3. Simulation Framework: Evaluate strategy performance without human input
"""

import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple, Set
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class Oracle:
    """
    Simulates a perfect human labeler with access to ground truth.

    Used for:
    - Automated testing of selection strategies
    - Performance benchmarking
    - Active learning research
    """

    def __init__(self, dataset, noise_rate: float = 0.0):
        """
        Args:
            dataset: MNISTWrapper or similar dataset with ground truth labels
            noise_rate: Probability of returning incorrect label (simulates human error)
        """
        self.dataset = dataset
        self.noise_rate = noise_rate
        self.query_count = 0
        self.history: List[Tuple[str, str]] = []  # (image_id, label)

    def label(self, image_id: str) -> str:
        """Get ground truth label for an image (with optional noise)."""
        self.query_count += 1

        idx = self.dataset.get_idx_by_id(image_id)
        if idx is None:
            raise ValueError(f"Image ID {image_id} not found in dataset")

        true_label = str(self.dataset.labels[idx])

        # Simulate labeling error
        if np.random.random() < self.noise_rate:
            # Return random wrong label
            all_labels = list(set(str(l) for l in self.dataset.labels))
            all_labels.remove(true_label)
            label = np.random.choice(all_labels)
        else:
            label = true_label

        self.history.append((image_id, label))
        return label

    def verify(self, image_id: str, predicted_label: str) -> bool:
        """Check if predicted label matches ground truth."""
        idx = self.dataset.get_idx_by_id(image_id)
        true_label = str(self.dataset.labels[idx])
        return predicted_label == true_label

    def get_stats(self) -> Dict:
        """Return oracle statistics."""
        return {
            "total_queries": self.query_count,
            "history_length": len(self.history),
            "noise_rate": self.noise_rate,
        }


class SelectionStrategy(ABC):
    """Base class for image selection strategies."""

    def __init__(self, name: str):
        self.name = name
        self.selection_history: List[str] = []

    @abstractmethod
    def select_next(
        self,
        model,
        dataset,
        qdrant_client,
        collection_name: str,
        user_labels: Dict[str, str],
        last_labeled_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Select the next image to label.

        Args:
            model: The embedding model
            dataset: Dataset wrapper
            qdrant_client: Qdrant client for vector search
            collection_name: Name of Qdrant collection
            user_labels: Dict of already labeled images {id: label}
            last_labeled_id: ID of the most recently labeled image

        Returns:
            image_id to label next, or None if all labeled
        """
        pass

    def reset(self):
        """Reset strategy state."""
        self.selection_history = []


class RandomStrategy(SelectionStrategy):
    """Randomly select unlabeled images."""

    def __init__(self):
        super().__init__("random")

    def select_next(
        self,
        model,
        dataset,
        qdrant_client,
        collection_name,
        user_labels,
        last_labeled_id=None,
    ):
        unlabeled_ids = [id for id in dataset.ids if id not in user_labels]
        if not unlabeled_ids:
            return None
        selected = np.random.choice(unlabeled_ids)
        self.selection_history.append(selected)
        return selected


class ClusterChainStrategy(SelectionStrategy):
    """
    Chain through clusters by always selecting the nearest unlabeled neighbor.

    This is the default QuickSort strategy - creates a "chain reaction" through
    similar images for efficient labeling.
    """

    def __init__(self, k_neighbors: int = 100):
        super().__init__("cluster_chain")
        self.k_neighbors = k_neighbors

    def select_next(
        self,
        model,
        dataset,
        qdrant_client,
        collection_name,
        user_labels,
        last_labeled_id=None,
    ):
        candidate_id = None

        # If we have a last labeled image, find its nearest unlabeled neighbor
        if last_labeled_id:
            idx_last = dataset.get_idx_by_id(last_labeled_id)
            if idx_last is not None:
                t_last = dataset.get_tensor(idx_last).unsqueeze(0)
                with torch.no_grad():
                    emb_last = model(t_last).cpu().squeeze().numpy()

                # Find neighbors
                hits = qdrant_client.query_points(
                    collection_name=collection_name,
                    query=emb_last,
                    limit=self.k_neighbors,
                ).points

                # Filter for unlabeled
                unlabeled_hits = [h for h in hits if h.id not in user_labels]

                if unlabeled_hits:
                    candidate_id = unlabeled_hits[0].id

        # Fallback to random if no chain candidate
        if candidate_id is None:
            candidate_id = dataset.get_unlabelled_id()

        if candidate_id:
            self.selection_history.append(candidate_id)

        return candidate_id


class UncertaintyStrategy(SelectionStrategy):
    """
    Select images where the model is most uncertain.

    Uses entropy of neighbor label distribution as uncertainty measure.
    """

    def __init__(self, k_neighbors: int = 50):
        super().__init__("uncertainty")
        self.k_neighbors = k_neighbors

    def select_next(
        self,
        model,
        dataset,
        qdrant_client,
        collection_name,
        user_labels,
        last_labeled_id=None,
    ):
        unlabeled_ids = [id for id in dataset.ids if id not in user_labels]
        if not unlabeled_ids:
            return None

        # Calculate uncertainty for a sample of unlabeled images
        # (sampling for efficiency - full scan would be slow)
        sample_size = min(100, len(unlabeled_ids))
        sample_ids = np.random.choice(unlabeled_ids, sample_size, replace=False)

        max_entropy = -1
        best_id = None

        model.eval()
        for img_id in sample_ids:
            idx = dataset.get_idx_by_id(img_id)
            tensor = dataset.get_tensor(idx).unsqueeze(0)

            with torch.no_grad():
                emb = model(tensor).cpu().squeeze().numpy()

            # Get neighbors
            hits = qdrant_client.query_points(
                collection_name=collection_name, query=emb, limit=self.k_neighbors
            ).points

            # Collect neighbor labels
            neighbor_labels = []
            for h in hits:
                if h.id != img_id and h.id in user_labels:
                    neighbor_labels.append(user_labels[h.id])

            if not neighbor_labels:
                # No labeled neighbors - high uncertainty
                entropy = 1.0
            else:
                # Calculate entropy
                label_counts = Counter(neighbor_labels)
                total = len(neighbor_labels)
                probs = np.array([count / total for count in label_counts.values()])
                entropy = -np.sum(probs * np.log(probs + 1e-10))

            if entropy > max_entropy:
                max_entropy = entropy
                best_id = img_id

        if best_id:
            self.selection_history.append(best_id)

        return best_id


class MarginStrategy(SelectionStrategy):
    """
    Select images at cluster boundaries (diverse/hard examples).

    Finds images whose nearest neighbors have mixed labels.
    """

    def __init__(self, k_neighbors: int = 20):
        super().__init__("margin")
        self.k_neighbors = k_neighbors

    def select_next(
        self,
        model,
        dataset,
        qdrant_client,
        collection_name,
        user_labels,
        last_labeled_id=None,
    ):
        unlabeled_ids = [id for id in dataset.ids if id not in user_labels]
        if not unlabeled_ids:
            return None

        # Sample for efficiency
        sample_size = min(100, len(unlabeled_ids))
        sample_ids = np.random.choice(unlabeled_ids, sample_size, replace=False)

        max_diversity = -1
        best_id = None

        model.eval()
        for img_id in sample_ids:
            idx = dataset.get_idx_by_id(img_id)
            tensor = dataset.get_tensor(idx).unsqueeze(0)

            with torch.no_grad():
                emb = model(tensor).cpu().squeeze().numpy()

            # Get neighbors
            hits = qdrant_client.query_points(
                collection_name=collection_name, query=emb, limit=self.k_neighbors
            ).points

            # Count unique labels among neighbors
            neighbor_labels = set()
            for h in hits:
                if h.id != img_id and h.id in user_labels:
                    neighbor_labels.add(user_labels[h.id])

            diversity = len(neighbor_labels)

            if diversity > max_diversity:
                max_diversity = diversity
                best_id = img_id

        if best_id:
            self.selection_history.append(best_id)

        return best_id


class DiversityStrategy(SelectionStrategy):
    """
    Select diverse images to maximize coverage of embedding space.

    Uses farthest-point sampling from already labeled images.
    """

    def __init__(self):
        super().__init__("diversity")

    def select_next(
        self,
        model,
        dataset,
        qdrant_client,
        collection_name,
        user_labels,
        last_labeled_id=None,
    ):
        unlabeled_ids = [id for id in dataset.ids if id not in user_labels]
        if not unlabeled_ids:
            return None

        if not user_labels:
            # No labels yet - random selection
            selected = np.random.choice(unlabeled_ids)
            self.selection_history.append(selected)
            return selected

        # Find unlabeled image farthest from all labeled images
        model.eval()

        # Get embeddings of all labeled images
        labeled_embeddings = []
        for labeled_id in user_labels.keys():
            idx = dataset.get_idx_by_id(labeled_id)
            tensor = dataset.get_tensor(idx).unsqueeze(0)
            with torch.no_grad():
                emb = model(tensor).cpu().squeeze().numpy()
            labeled_embeddings.append(emb)

        labeled_embeddings = np.array(labeled_embeddings)

        # Sample unlabeled for efficiency
        sample_size = min(100, len(unlabeled_ids))
        sample_ids = np.random.choice(unlabeled_ids, sample_size, replace=False)

        max_min_dist = -1
        best_id = None

        for img_id in sample_ids:
            idx = dataset.get_idx_by_id(img_id)
            tensor = dataset.get_tensor(idx).unsqueeze(0)

            with torch.no_grad():
                emb = model(tensor).cpu().squeeze().numpy()

            # Calculate minimum distance to any labeled image
            dists = np.linalg.norm(labeled_embeddings - emb, axis=1)
            min_dist = np.min(dists)

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_id = img_id

        if best_id:
            self.selection_history.append(best_id)

        return best_id


# Strategy registry
STRATEGIES = {
    "random": RandomStrategy,
    "cluster_chain": ClusterChainStrategy,
    "uncertainty": UncertaintyStrategy,
    "margin": MarginStrategy,
    "diversity": DiversityStrategy,
}


def get_strategy(name: str, **kwargs) -> SelectionStrategy:
    """Factory function to create selection strategies."""
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy: {name}. Available: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[name](**kwargs)
