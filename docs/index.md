# QuickSort: High-Performance Image Clustering & Labelling

**QuickSort** is a modern, high-speed interactive tool for labelling and clustering image datasets. It leverages deep learning embeddings, vector search (Qdrant), and dimensionality reduction (UMAP via `squeeze`) to create a fluid "Active Learning" experience.

![Dashboard Preview](https://placehold.co/800x400/121214/white?text=QuickSort+Dashboard)

## Core Philosophy: Flow State Labelling

QuickSort is designed to minimize friction. Traditional labelling tools are click-heavy and slow. QuickSort introduces a **Blended Predictive Workflow**:

1.  **Context-Aware Suggestions**: The system uses the embedding space to find neighbors of what you just labelled.
2.  **Chain Reaction**: Label one "Cat", and the system immediately pulls up similar "Cats" and suggests the label "Cat". You just press `Shift + Enter` to confirm.
3.  **Global Visualization**: A real-time 2D scatter plot (Squeeze UMAP) shows your progress and the dataset's structure.

## Key Features

- **⚡️ Blended Workflow**: Seamlessly switch between accepting suggestions (`J` / `Shift+Enter`) and manual typing/correction.
- **🧠 Active Learning**: The system learns from your labels in real-time, refining its neighbor search to present the most relevant next samples.
- **⏮️ History & Correction**: Made a mistake? Hit `Alt + Left` to step back in time, correct a label, and watch the embedding map update instantly.
- **🎨 Modern UI**: A dark-themed, glassmorphism-inspired interface built for long sessions without eye strain.
- **🗺️ Embedding Map**: Interactive WebGL scatter plot (`regl-scatterplot`) visualizing 1000s of data points.

## Getting Started

Check out the [Usage Guide](usage.md) to master the keyboard shortcuts, or dive into the [Architecture](architecture.md) to understand how it works under the hood.
