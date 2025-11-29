# API Reference

## Endpoints

### `GET /pair`

Returns a pair of images for comparison.

**Response:**

```json
{
  "image1": {
    "id": "uuid-string",
    "data": "base64-encoded-image"
  },
  "image2": {
    "id": "uuid-string",
    "data": "base64-encoded-image"
  },
  "debug_strategy": "random|hard"
}
```

### `POST /vote`

Submits user feedback for a pair of images.

**Request Body:**

```json
{
  "id1": "uuid-string",
  "id2": "uuid-string",
  "are_same": true
}
```

**Response:**

```json
{
  "status": "ok",
  "buffer_size": 1
}
```
