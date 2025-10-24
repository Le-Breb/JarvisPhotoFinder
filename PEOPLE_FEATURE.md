# People Management Feature

This document explains the new people management feature that allows you to browse, manage, and correct face recognition results.

## Features

### 1. Face Clustering
- Automatically groups similar faces together using DBSCAN clustering
- Creates "people" from face clusters
- Handles outliers (faces that don't match any person)

### 2. People Page (`/people`)
- Grid view of all identified people
- Shows representative photo for each person
- Displays photo count per person
- Click to view person details

### 3. Person Detail Page (`/people/[id]`)
- View all photos of a specific person
- **Rename people**: Edit the person's name
- **Remove incorrect faces**: Mark photos where the person is misidentified
- Automatic cleanup: If a person has fewer than 2 photos, they're moved to "unmatched faces"

### 4. Navigation
- New "People" link in header navigation
- Easy access between Search and People pages

## Setup Instructions

### 1. Install Python Dependencies

First, install the required Python package for clustering:

```bash
cd website/python
pip install scikit-learn
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### 2. Index Faces

Build the face embeddings database:

```bash
python index_faces.py index
```

This will:
- Scan all images in the `images/` folder
- Detect faces using InsightFace
- Save face embeddings, filenames, and bounding boxes

### 3. Cluster Faces

Group similar faces into people:

```bash
python index_faces.py cluster
```

This will:
- Load face embeddings
- Use DBSCAN to cluster similar faces
- Create `faces/clusters.json` with person data
- Assign default names like "Person 0", "Person 1", etc.

### 4. View People

```bash
# List all people in terminal
python index_faces.py list

# Rename a person
python index_faces.py rename 0 "John Doe"
```

### 5. Use the Web Interface

1. Start the Next.js development server:
   ```bash
   cd website
   npm run dev
   ```

2. Navigate to `http://localhost:3000/people`

3. Browse and manage people:
   - Click on a person to view all their photos
   - Click "Rename" to change their name
   - Hover over photos and click "Not [Name]" to remove incorrect identifications

## API Endpoints

### GET `/api/people`
List all people with representative photos.

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
Get detailed information about a specific person.

**Response:**
```json
{
  "person": {
    "cluster_id": "0",
    "name": "Person 0",
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

**Request:**
```json
{
  "name": "John Doe"
}
```

**Response:**
```json
{
  "success": true,
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

If the person has fewer than 2 faces remaining, they're removed:
```json
{
  "success": true,
  "remainingFaces": 1,
  "movedToNoise": true
}
```

## File Structure

```
website/
├── app/
│   ├── people/
│   │   ├── page.tsx              # People grid view
│   │   └── [id]/
│   │       └── page.tsx          # Person detail page
│   └── api/
│       └── people/
│           ├── route.ts          # List all people
│           └── [id]/
│               ├── route.ts      # Get/update person
│               └── faces/
│                   └── [faceId]/
│                       └── route.ts  # Remove face
├── components/
│   └── header.tsx                # Updated with People link
├── types/
│   └── index.ts                  # Person, Face types
└── python/
    ├── index_faces.py            # Face indexing & clustering
    └── faces/
        ├── embeddings.npy        # Face embeddings
        ├── filenames.npy         # Image filenames
        ├── bboxes.npy           # Face bounding boxes
        └── clusters.json         # Person clusters (editable)
```

## Clustering Parameters

You can adjust clustering behavior in `index_faces.py`:

```python
cluster_faces(
    eps=0.5,        # Distance threshold (lower = stricter)
    min_samples=2   # Minimum faces to form a cluster
)
```

- **eps** (0.4-0.6): Lower values create more clusters (stricter matching)
- **min_samples** (2-5): Minimum faces needed to form a person

## Troubleshooting

### No people found
1. Run `python index_faces.py index` first
2. Then run `python index_faces.py cluster`
3. Check that `faces/clusters.json` exists

### Too many/few clusters
Adjust the `eps` parameter:
- Too many clusters → Increase eps (e.g., 0.6)
- Too few clusters → Decrease eps (e.g., 0.4)

### Person not appearing after changes
The web UI reads from `faces/clusters.json`. Make sure your Python scripts save changes there.

## Future Enhancements

Possible improvements:
- [ ] Merge two people clusters (same person identified twice)
- [ ] Search by face (upload photo to find that person)
- [ ] Face tagging in search results
- [ ] Bulk operations (select multiple faces to remove)
- [ ] Manual face annotation (add faces to a person)
