# Architecture

## System Design

The system is designed to be simple yet effective for extreme classification tasks.

### Components

1.  **Embedding Network**: A Convolutional Neural Network (CNN) that maps images to a 64-dimensional vector space.
2.  **Qdrant Vector Database**: Stores the embeddings and allows for fast similarity search.
3.  **Active Learning Loop**:
    *   The system proposes pairs of images.
    *   User votes "Same" or "Different".
    *   Feedback is used to compute a Contrastive Loss.
    *   The model is updated via backpropagation.
    *   Embeddings in Qdrant are refreshed.

### Data Flow

1.  **Initialization**: Synthetic images are generated and indexed in Qdrant.
2.  **Query**: The frontend requests a pair of images.
3.  **Selection Strategy**:
    *   **Random**: Two random images are selected.
    *   **Hard Mining**: An image is selected, and Qdrant finds its nearest neighbors to present a "hard" pair (ambiguous or close in vector space).
4.  **Feedback**: User submits a vote.
5.  **Training**: After a batch of votes, the model is trained, and the index is updated.
