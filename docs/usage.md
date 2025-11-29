# Usage Guide

## Setup

1.  Ensure you have Python 3.12+ installed.
2.  Install dependencies:
    ```bash
    pip install -r backend/requirements.txt
    ```
3.  Start the application:
    ```bash
    ./start.sh
    ```

## Sorting Images

1.  Open the web interface.
2.  You will see two images side-by-side.
3.  Determine if they belong to the same category.
    *   Click **YES** (or press `Left Arrow` / `Y`) if they are the same.
    *   Click **NO** (or press `Right Arrow` / `N`) if they are different.
4.  The system will learn from your feedback and present more challenging pairs over time.
