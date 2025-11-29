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
*   **Accept Suggestion**: Press **`J`** or **`Shift + Enter`**.
*   **Reject/Correct**:
    *   Press **`I`** to focus the input (if not focused).
    *   Just start typing the correct label to override.
    *   Press `Enter` to submit the correction.

### History & Correction
*   **Navigate Back**: Press **`Alt + Left Arrow`** to view previous labels.
*   **Correct Mistake**: Type the new label and press `Enter`.
*   **Navigate Forward**: Press **`Alt + Right Arrow`** to return to the live queue.

## Keyboard Shortcuts Cheat Sheet

| Action | Shortcut | Context |
| :--- | :--- | :--- |
| **Accept Suggestion** | `J` or `Shift + Enter` | When suggestion is visible |
| **Reject Suggestion** | `I` | Focus input to correct |
| **Submit Label** | `Enter` | Input mode |
| **Autocomplete Select** | `Up` / `Down` Arrows | Input mode |
| **Autocomplete Fill** | `Tab` | Input mode |
| **History Back** | `Alt + Left` | Global |
| **History Forward** | `Alt + Right` | Global |
| **Toggle Scatter Plot** | Click Icon | UI Toolbar |
| **Toggle Logs** | Click Icon | UI Toolbar |
