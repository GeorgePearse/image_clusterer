# API Reference

## Backend Endpoints

The backend runs on port `8000` by default.

### `GET /points`

Returns the full dataset as 2D coordinates for the scatter plot.

**Response:**
```json
[
  {
    "id": "uuid-1",
    "x": 12.5,
    "y": -4.2,
    "label": "cat"
  },
  ...
]
```

### `GET /next`

Retrieves the next image to label, along with a predictive suggestion.

**Response:**
```json
{
  "image": {
    "id": "uuid-string",
    "data": "base64-encoded-png..."
  },
  "suggestion": "cat", // null if no suggestion
  "debug_info": {
    "neighbor_count": 5,
    "neighbors": ["cat", "cat", "dog"]
  }
}
```

### `POST /label`

Submits a label for an image.

**Request:**
```json
{
  "image_id": "uuid-string",
  "label": "cat"
}
```

**Response:**
```json
{
  "status": "ok",
  "buffer_size": 1,
  "training": false
}
```

### `POST /vote`

(Legacy/Background) Submits a pair comparison vote for metric learning.

**Request:**
```json
{
  "id1": "uuid-string",
  "id2": "uuid-string",
  "are_same": true
}
```
