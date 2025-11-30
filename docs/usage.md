# Usage Guide

QuickSort is built for keyboard-first usage. Master these shortcuts to reach maximum labelling speed.

## Quick Start

1.  **Start the System**:
    ```bash
    ./start.sh
    ```
2.  **Open Dashboard**: Navigate to `http://localhost:5173`.

## The Workflow

### Phase 1: Bootstrap (Teaching)
*   **State**: The system has no labels. It shows random images.
*   **Action**: Type the label (e.g., "one", "cat") and press `Enter`.
*   **Autocomplete**: As you type, suggestions appear. Use `Arrow Keys` to select and `Enter` to confirm.

### Phase 2: Active Learning (Flow State)
*   **State**: The system suggests a label (e.g., "Is this a **one**?").
*   **Accept Suggestion**: Press **`Shift + Enter`**.
*   **Reject/Correct**:
    *   Press **`I`** to focus the input (if not focused).
    *   Just start typing the correct label to override.
    *   Press `Enter` to submit the correction.

### History & Correction
*   **Navigate Back**: Press **`Shift + Left Arrow`** to view previous labels.
*   **Correct Mistake**: Type the new label and press `Enter`.
*   **Navigate Forward**: Press **`Shift + Right Arrow`** to return to the live queue.

### Skip / "Don't Know"
*   **Quick Skip**: Press **`Shift + ?`** to mark an image as "don't know".
*   **Purpose**: Creates a queue of ambiguous images for later review.
*   **Behavior**: Images marked "don't know" are stored but **never** suggested as predictions. The system won't offer "don't know" as an AI suggestion, keeping your flow state intact.
*   **Review Later**: Use history navigation (`Shift + Left/Right`) to revisit and re-label skipped images when you have more context.

## Keyboard Shortcuts Cheat Sheet

| Action | Shortcut | Context |
| :--- | :--- | :--- |
| **Accept Suggestion** | `Shift + Enter` | When suggestion is visible |
| **Skip / Don't Know** | `Shift + ?` | Global |
| **Reject Suggestion** | `I` | Focus input to correct |
| **Submit Label** | `Enter` | Input mode |
| **Autocomplete Select** | `Up` / `Down` Arrows | Input mode |
| **Autocomplete Fill** | `Tab` | Input mode |
| **Quick Select** | `1` - `9` | Select suggestion 1-9 immediately |
| **History Back** | `Shift + Left` | Global |
| **History Forward** | `Shift + Right` | Global |
| **Toggle Scatter Plot** | Click Icon | UI Toolbar |
| **Toggle Logs** | Click Icon | UI Toolbar |
| **Toggle Simulation** | Click Icon | UI Toolbar |

## Advanced Controls

### Strategy Selection
In the top header, you can switch the active learning strategy using the **Strategy Dropdown** (Lightning icon).
*   **Cluster Chain**: Default. Prioritizes consistency and speed (stay in the same class).
*   **Uncertainty**: Selects images the model is least sure about. Good for boundary refinement.
*   **Random**: Pure random sampling. Good for initial exploration or benchmarking.

### Simulation View
Click the **Bar Chart** icon in the header to open the Simulation View.
*   Run background simulations to compare strategies.
*   Visualize model accuracy improvement over time.
*   See the impact of "Flow State" logic vs pure information gain.
