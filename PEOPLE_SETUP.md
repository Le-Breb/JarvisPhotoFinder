# People Feature Setup Guide

This guide explains how to set up and use the face recognition and people management feature in JarvisPhotoFinder.

## Overview

The people feature allows you to:
- Automatically detect and cluster faces in your photos
- Assign names to people
- Browse all photos of a specific person
- Remove incorrectly identified faces
- View photos in full-screen mode

## Prerequisites

1. Python 3.8+ installed
2. Images stored in `website/python/images/` directory

## Installation

### 1. Install Python Dependencies

```bash
cd website/python
pip install -r requirements.txt
```

Required packages:
- `insightface` - Face detection and recognition
- `scikit-learn` - Clustering algorithm (DBSCAN)
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `tqdm` - Progress bars

### 2. Download Face Recognition Model

The first time you run the face indexing, InsightFace will automatically download the `buffalo_l` model (~600MB).

## Usage

### Step 1: Index Faces

Extract and embed all faces from your images:

```bash
cd website/python
python index_faces.py index
```

This will:
- Scan all images in the `images/` folder
- Detect faces in each image
- Generate embeddings (face representations)
- Save to `faces/embeddings.npy`, `faces/filenames.npy`, and `faces/bboxes.npy`

### Step 2: Cluster Faces

Group similar faces together automatically:

```bash
python index_faces.py cluster
```

This will:
- Load face embeddings
- Use DBSCAN clustering with cosine similarity
- Group faces of the same person
- Save clusters to `faces/clusters.json` and `faces/clusters.pkl`

**Clustering Parameters** (in `index_faces.py`):
- `eps=0.5` - Maximum distance for clustering (lower = stricter matching)
- `min_samples=2` - Minimum faces to form a cluster

Adjust these if you get too many or too few clusters.

### Step 3: View People in Web UI

1. Start your Next.js development server:
```bash
cd website
npm run dev
```

2. Navigate to **People** page from the header menu

3. You'll see all detected people with their representative photo

## Web Interface Features

### People Page (`/people`)

- **Grid View**: Shows all identified people
- **Quick Rename**: Hover over a person card and click the pencil icon to rename
- **Photo Count**: Displays number of photos for each person
- **Click to View**: Click any person card to see all their photos

### Person Detail Page (`/people/[id]`)

- **Full Photo Gallery**: All photos containing that person
- **Rename Person**: Click "Rename" button to change the person's name
- **Image Viewer**: Click any photo to open in full-screen viewer
  - Arrow keys to navigate between photos
  - ESC key to close
- **Remove Incorrect Faces**: 
  - Hover over a photo
  - Click the trash icon
  - Confirm removal in dialog
  - If cluster becomes too small (<2 faces), the person is removed

## API Endpoints

### GET `/api/people`
Returns list of all people with summary info.

**Response:**
```json
{
  "people": [
    {
      "id": "0",
      "name": "Person 0",
      "faceCount": 15,
      "representative": {
        "face_id": "42",
        "filename": "images/photo.jpg",
        "bbox": [x, y, w, h]
      }
    }
  ]
}
```

### GET `/api/people/[id]`
Returns detailed information for a specific person.

**Response:**
```json
{
  "person": {
    "cluster_id": "0",
    "name": "John Doe",
    "faces": [
      {
        "face_id": "42",
        "filename": "images/photo.jpg",
        "bbox": [x, y, w, h],
        "embedding_idx": 42
      }
    ]
  }
}
```

### PUT `/api/people/[id]`
Update a person's name.

**Request Body:**
```json
{
  "name": "John Doe"
}
```

### DELETE `/api/people/[id]/faces/[faceId]`
Remove an incorrectly identified face from a person's cluster.

**Response:**
```json
{
  "success": true,
  "remainingFaces": 14,
  "movedToNoise": false
}
```

## Python CLI Commands

### List All People
```bash
python index_faces.py list
```

### Rename a Person
```bash
python index_faces.py rename <cluster_id> "New Name"
```

Example:
```bash
python index_faces.py rename 0 "John Doe"
```

### Search for a Person
```bash
python index_faces.py search path/to/person_photo.jpg
```

Returns similar faces ranked by similarity score.

## File Structure

```
website/python/
├── index_faces.py          # Face indexing and clustering
├── images/                 # Your photos (input)
└── faces/                  # Generated data (output)
    ├── embeddings.npy      # Face embeddings
    ├── filenames.npy       # Source image paths
    ├── bboxes.npy          # Face bounding boxes
    ├── clusters.json       # Cluster data (for web)
    └── clusters.pkl        # Cluster data (for Python)
```

## Troubleshooting

### No faces detected
- Ensure images have clear, front-facing faces
- Check image quality and resolution
- Try different lighting conditions

### Too many clusters (same person split)
- Increase `eps` parameter (e.g., `eps=0.6`)
- Re-run clustering: `python index_faces.py cluster`

### Too few clusters (different people merged)
- Decrease `eps` parameter (e.g., `eps=0.4`)
- Increase `min_samples` (e.g., `min_samples=3`)
- Re-run clustering

### CUDA/GPU errors
- The script will automatically fall back to CPU
- For faster processing, ensure CUDA is properly installed

### Import errors
- Make sure all dependencies are installed:
  ```bash
  pip install -r requirements.txt
  ```

## Performance Tips

1. **GPU Acceleration**: If you have an NVIDIA GPU with CUDA:
   - Install `onnxruntime-gpu` instead of `onnxruntime`
   - Face detection will be significantly faster

2. **Batch Processing**: Process images in batches if you have thousands of photos

3. **Image Preprocessing**: Compress large images before indexing to speed up processing

4. **Re-clustering**: You only need to re-run clustering if you adjust parameters, not re-index

## Advanced Usage

### Custom Clustering Parameters

Edit `index_faces.py` and modify the `cluster_faces()` function call:

```python
cluster_faces(
    eps=0.45,           # Stricter matching
    min_samples=3       # Need at least 3 faces
)
```

### Merge Clusters Programmatically

If the same person was split into multiple clusters, use Python:

```python
from index_faces import merge_clusters

merge_clusters(
    cluster_ids=[0, 3, 7],
    new_name="John Doe"
)
```

## Security Notes

- Face data is stored locally only
- No data is sent to external services
- Embeddings are mathematical representations, not raw face images
- Original photos remain in your `images/` folder

## Next Steps

1. Index your photos: `python index_faces.py index`
2. Cluster faces: `python index_faces.py cluster`
3. Open web UI and go to `/people`
4. Rename people and correct any mistakes
5. Enjoy browsing photos by person!
