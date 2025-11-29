# Product Design & Philosophy

QuickSort is not just a labeling tool; it is a **flow-state engine** designed to maximize the speed and enjoyment of data curation. Every interaction is crafted to reduce cognitive load and physical latency, allowing the user to label at the speed of thought.

## Core Philosophy

### 1. Speed is a Feature
Labeling data is traditionally a tedious, low-latency task. QuickSort treats latency as the enemy.
*   **Zero-Latency Interactions**: Transitions between images are instant.
*   **Optimized Shortcuts**: Common actions require zero mouse movement.
*   **Predictive Pre-fetching**: The next image is decided and fetched before the current one is finished.

### 2. Flow State
The interface is designed to keep the user in "the zone".
*   **Chain Reaction**: Instead of random sampling, QuickSort presents a stream of similar images. Once you start labeling "cats", you will keep seeing "cats" until that cluster is exhausted. This reduces context switching overhead.
*   **Visual Rhythm**: The UI provides subtle visual cues (glows, pulses) to maintain a rhythmic cadence without being distracting.

## Key Features

### The "Chain Reaction" Loop

The heart of QuickSort is the active learning loop that prioritizes **consistency over randomness**.

1.  **Trigger**: You label an image (e.g., "Golden Retriever").
2.  **Search**: The system instantly finds the nearest *unlabeled* neighbor in the embedding space.
3.  **Suggestion**: It sees that the neighbor is surrounded by other "Golden Retrievers".
4.  **Presentation**: The next image appears with a high-confidence suggestion: *"Is it Golden Retriever?"*
5.  **Action**: You simply press `Shift + Enter` to confirm.

This turns a complex decision ("What class is this?") into a binary reflex ("Is this prediction correct?"), which is significantly faster for the human brain to process.

### Keyboard-First Control

To maximize speed, hands should never leave the keyboard.

*   **`Shift + Enter` (The "Yes" Key)**: Confirms the top suggestion. This is the primary interaction during a Chain Reaction.
*   **`1` - `9` (Rapid Select)**: Instantly selects one of the top predictions.
    *   The suggestions are dynamically ordered by the model's confidence (KNN vote).
    *   If the model is 90% sure it's a "Cat" and 10% "Dog", pressing `1` selects "Cat" and `2` selects "Dog".
*   **`Shift + ← / →` (History)**: Quickly review and correct previous labels without breaking flow.

### Intelligent Suggestions

The suggestion system is more than just a classifier output. It uses a **K-Nearest Neighbors (KNN)** voting mechanism on the fly:

1.  The system looks at the `K` nearest labeled images to the current candidate.
2.  It aggregates their labels.
3.  It presents a sorted list of likely classes.

This means the system **learns in real-time**. If you introduce a new class "Astronaut" and label a few examples, the very next similar image will have "Astronaut" as the top suggestion (mapped to key `1`).

## Visual Feedback

*   **Embedding Space (Scatter Plot)**: A live view of the model's brain. As you label, you see clusters forming and shifting. This provides global context and trust in the model's learning process.
*   **Confidence Glow**: When the model is confident in a suggestion, the UI glows with the suggestion's color, subtly reinforcing the "happy path" of just pressing `Shift + Enter`.
