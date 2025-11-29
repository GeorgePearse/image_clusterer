# Roadmap

## Future Directions

QuickSort is evolving from a single-label classification tool into a comprehensive data curation platform. Our next major milestone involves supporting rich, natural language descriptions to train Vision-Language Models (VLMs).

### 🚀 Full-Text Descriptions & CLIP Support

We plan to move beyond simple categorical labels (e.g., "cat", "dog") to support dense captions and full-text descriptions (e.g., "a golden retriever sitting on a porch at sunset"). This will enable the fine-tuning of models like **CLIP**, **SigLIP**, and other multimodal architectures.

#### Planned Features

1.  **Rich Text Input Interface**
    *   Replace the single-line input with a multi-line text area for detailed captioning.
    *   Support for "tags" + "description" hybrid workflows.

2.  **VLM-Assisted Labelling**
    *   Integrate a small, fast VLM (e.g., Llava, PaliGemma) on the backend to *generate* initial caption suggestions.
    *   User workflow shifts to "Review & Edit" rather than writing from scratch.

3.  **Text-Image Contrastive Learning**
    *   Update the training loop to support InfoNCE loss between image embeddings and text embeddings.
    *   Allow the system to learn semantic nuances (e.g., distinguishing "shiny red car" from "matte red car") rather than just broad classes.

4.  **Semantic Search Integration**
    *   Use the text descriptions to index images in Qdrant.
    *   Enable natural language search within the tool (e.g., "find all images that look blurry").

### 🧠 VLM In-Context Learning (Long-Term)

The ultimate goal is to transition from training external embedding models to leveraging **Vision-Language Models (VLMs)** directly via **In-Context Learning (ICL)**.

Instead of fine-tuning weights (which is slow and compute-heavy), we will update the system's "knowledge" by dynamically constructing prompts that include the user's recently labeled images.

#### The Vision
1.  **Dynamic Context Construction**: When the user labels an image "A", that image-label pair is added to a retrieval bank.
2.  **Few-Shot Prompting**: For the next image "B", we retrieve the most similar labeled examples (e.g., "A") and insert them into the VLM's context window.
    *   *Prompt:* "Here is an image of a cat [Image A]. What is this image [Image B]?"
3.  **Instant Adaptation**: The model "learns" the new class immediately without a gradient update.

#### Current Challenges & Strategy
*   **Latency**: Sending multiple images + prompt to a large VLM (e.g., GPT-4o, Gemini 1.5) for every single label action is currently too slow for the "flow state" experience we target (< 200ms).
*   **Cost**: High token/image costs for continuous usage.

**Mitigation Plan**:
*   **Hybrid Approach**: Use fast embeddings (CLIP/SigLIP) for the real-time "Chain Reaction" loop and Qdrant retrieval.
*   **Async VLM**: Use the VLM only for difficult cases or to periodically "clean up" and verify clusters in the background.
*   **Small VLMs**: Experiment with local, quantized VLMs (e.g., LLaVA-Next, PaliGemma 3B) that can run at higher frame rates on consumer GPUs.

### 📦 COCO Integration & Data Pipeline (Rust)

A major workflow improvement will be seamless integration with existing annotation formats. We plan to build a high-performance **Rust package** for COCO file processing:

#### Features
1.  **COCO Import Pipeline**
    *   Parse COCO JSON files (annotations or model predictions)
    *   Extract image crops based on bounding boxes
    *   Convert detections/segments into individual classification tasks
    *   Feed directly into QuickSort's labeling workflow

2.  **Detection → Classification Bridge**
    *   Use existing object detector (YOLO, RCNN, etc.) to generate proposals
    *   Import COCO predictions as unlabeled crop candidates
    *   Apply QuickSort's active learning to verify/correct class labels
    *   Export cleaned annotations back to COCO format

3.  **Performance Benefits of Rust**
    *   **Fast parsing**: Handle massive COCO files (100K+ images) in seconds
    *   **Zero-copy deserialization**: Efficiently process large JSON without Python overhead
    *   **Parallel crop extraction**: Multi-threaded image processing
    *   **Memory efficiency**: Stream processing for datasets larger than RAM

#### Example Workflow
```bash
# Import COCO predictions and start labeling
quicksort-coco import \
    --predictions model_output.json \
    --images /path/to/coco/images \
    --output-dir crops/ \
    --min-confidence 0.3

# QuickSort processes the crops
./start.sh --dataset crops/

# Export corrected labels back to COCO
quicksort-coco export \
    --labels labels.json \
    --output cleaned_annotations.json
```

#### Integration Points
*   Python bindings via PyO3 for seamless backend integration
*   CLI tool for standalone pipeline usage
*   Support for COCO variants (Detectron2, MMDetection formats)
*   Optional GPU acceleration for crop extraction (image-rs + wgpu)

This addresses a critical gap: connecting QuickSort to existing CV pipelines where COCO is the standard format.

### 🛠 Technical Improvements

*   **Multi-User Support**: Websocket concurrency handling for team-based labelling.
*   **Plugin System**: Allow custom embedding models (ResNet, ViT, CLIP) to be swapped via configuration.
*   **Export Formats**: Native support for exporting to COCO, Parquet, and WebDataset formats.
