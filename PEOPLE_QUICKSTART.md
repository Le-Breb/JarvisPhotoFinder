# People Feature - Quick Start Guide

## 🚀 Getting Started in 3 Steps

### Step 1: Index Your Photos
```bash
cd website/python
python index_faces.py index
```
**What it does:** Scans all images and detects faces
**Time:** ~1-2 minutes for 1000 photos

### Step 2: Cluster Faces
```bash
python index_faces.py cluster
```
**What it does:** Groups similar faces together
**Time:** ~10-30 seconds

### Step 3: Browse in Web UI
```bash
cd ../
npm run dev
```
Navigate to **People** in the header menu

---

## 📸 Feature Overview

### People Page (`/people`)
```
┌─────────────────────────────────────────────────┐
│  🏠 Semantic Search  |  🔍 Search  |  👥 People  │
├─────────────────────────────────────────────────┤
│                                                  │
│  People                                          │
│  Browse and manage identified people             │
│                                                  │
│  Found 15 people                                 │
│                                                  │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐       │
│  │ 👤  │ │ 👤  │ │ 👤  │ │ 👤  │ │ 👤  │       │
│  │     │ │     │ │     │ │     │ │     │       │
│  │John │ │Sarah│ │Mike │ │Lisa │ │Tom  │       │
│  │15   │ │23   │ │8    │ │12   │ │6    │       │
│  └─────┘ └─────┘ └─────┘ └─────┘ └─────┘       │
│                                                  │
└─────────────────────────────────────────────────┘

✏️  Hover over a person → Click pencil icon → Rename
🖱️  Click person card → View all their photos
```

### Person Detail Page (`/people/0`)
```
┌─────────────────────────────────────────────────┐
│  🏠 Semantic Search  |  🔍 Search  |  👥 People  │
├─────────────────────────────────────────────────┤
│                                                  │
│  ← Back to People                                │
│                                                  │
│  John Doe  ✏️ Rename                             │
│  15 photos                                       │
│                                                  │
│  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐       │
│  │       │ │       │ │       │ │       │       │
│  │ Photo │ │ Photo │ │ Photo │ │ Photo │       │
│  │   🗑️  │ │   🗑️  │ │   🗑️  │ │   🗑️  │       │
│  └───────┘ └───────┘ └───────┘ └───────┘       │
│                                                  │
└─────────────────────────────────────────────────┘

🖱️  Click photo → Full-screen viewer
🗑️  Hover → Click trash → Remove incorrect face
⌨️  Arrow keys → Navigate photos
⌨️  ESC → Close viewer
```

---

## 🎯 Common Tasks

### Rename a Person
**Option 1: Quick Rename (People Page)**
1. Hover over person card
2. Click pencil icon ✏️
3. Type new name
4. Press Enter or click ✓

**Option 2: Detail Page**
1. Click person card
2. Click "Rename" button
3. Type new name
4. Click "Save"

### Remove Wrong Face
1. Go to person's detail page
2. Hover over incorrect photo
3. Click trash icon 🗑️
4. Confirm in dialog
5. Face removed from cluster

### View Photos in Full Screen
1. Go to person's detail page
2. Click any photo
3. Use ← → arrow keys to navigate
4. Press ESC to close

---

## 🔧 Troubleshooting

### "No people found yet"
**Problem:** Face indexing hasn't been run
**Solution:**
```bash
cd website/python
python index_faces.py index
python index_faces.py cluster
```

### Same person split into multiple entries
**Problem:** Clustering too strict
**Solution:**
```bash
# Edit index_faces.py, change eps=0.5 to eps=0.6
# Then re-cluster:
python index_faces.py cluster
```

### Different people merged together
**Problem:** Clustering too loose
**Solution:**
```bash
# Edit index_faces.py, change eps=0.5 to eps=0.4
# Then re-cluster:
python index_faces.py cluster
```

### Person deleted after removing one photo
**Problem:** Cluster needs at least 2 faces
**Solution:** This is by design. If only 1 photo remains after removal, it's moved to "noise" (unclustered faces)

---

## 🎨 UI Features

### Interactive Elements
- ✏️ **Pencil Icon**: Quick rename
- 🗑️ **Trash Icon**: Remove face
- 🖱️ **Hover Effects**: Scale images, show controls
- ⌨️ **Keyboard Shortcuts**: Enter, ESC, Arrows

### Visual Feedback
- 📊 **Photo Count**: Shows number of photos per person
- 🔄 **Loading States**: Skeleton screens while loading
- ✅ **Success Messages**: Confirms actions
- ⚠️ **Error Messages**: Helpful error descriptions

### Responsive Design
- 📱 **Mobile**: 2 columns
- 💻 **Tablet**: 3-4 columns
- 🖥️ **Desktop**: 5-6 columns

---

## 💡 Pro Tips

1. **Better Detection**: Use photos with clear, front-facing faces
2. **Faster Indexing**: Use GPU if available (CUDA)
3. **Adjust Clustering**: Tweak `eps` parameter for your specific use case
4. **Re-index vs Re-cluster**: 
   - Only re-index when adding new photos
   - Re-cluster anytime to adjust grouping
5. **Backup**: Keep a copy of `faces/clusters.json` before major changes

---

## 📊 Data Flow

```
Photos (images/)
    ↓
Index Faces
    ↓
embeddings.npy, filenames.npy, bboxes.npy
    ↓
Cluster Faces
    ↓
clusters.json, clusters.pkl
    ↓
API Routes (/api/people/*)
    ↓
Web UI (/people, /people/[id])
    ↓
User Actions (rename, remove)
    ↓
Update clusters.json
```

---

## 📚 More Help

- **Detailed Setup**: See `PEOPLE_SETUP.md`
- **Implementation Details**: See `PEOPLE_FEATURE_SUMMARY.md`
- **Original Spec**: See `PEOPLE_FEATURE.md`

---

## ✅ Quick Checklist

Before using the feature:
- [ ] Photos in `website/python/images/`
- [ ] Python dependencies installed (`pip install -r requirements.txt`)
- [ ] Faces indexed (`python index_faces.py index`)
- [ ] Faces clustered (`python index_faces.py cluster`)
- [ ] Web server running (`npm run dev`)

You're ready to use the People feature! 🎉
