# Search by Person Name Feature

## Overview

When you search for a person's name in the search bar, the system now automatically detects if that name matches someone in your people database and switches to face-based search instead of text-based CLIP search.

## How It Works

### Search Flow

```
User types "John Doe" in search
    ↓
Backend checks clusters.json for name match
    ↓
If match found:
    ↓
Use representative face from that person's cluster
    ↓
Search all faces by similarity to representative
    ↓
Return photos ranked by face similarity
```

### Name Matching Logic

The system checks if your search query:
- **Exact match**: Query exactly matches a person name
- **Contains**: Query is contained in person name  
- **Partial**: Person name is contained in query

**Example matches:**
- Search: `"john"` → Matches: `"John Doe"`
- Search: `"john doe"` → Matches: `"John Doe"`
- Search: `"doe"` → Matches: `"John Doe"`
- Search: `"sarah smith"` → Matches: `"Sarah Smith"`

All matching is **case-insensitive**.

---

## Backend Implementation

### Key Functions

#### 1. `check_person_name_match(query)`
Checks if search query matches any person name in clusters.

```python
def check_person_name_match(query):
    """Check if query matches a person name in clusters"""
    if not people_clusters or not people_clusters.get('clusters'):
        return None
    
    query_lower = query.lower().strip()
    
    for cluster_id, cluster_data in people_clusters['clusters'].items():
        person_name = cluster_data.get('name', '').lower().strip()
        if person_name == query_lower or query_lower in person_name or person_name in query_lower:
            return cluster_data.get('name')
    
    return None
```

#### 2. `search_person_by_name(person_name, top_k=50)`
Searches for photos using face similarity based on person's cluster.

**Process:**
1. Find cluster matching the person name
2. Get representative embedding (first face in cluster)
3. Calculate distances to all faces in index
4. Return top K most similar faces
5. Avoid duplicate image files

```python
def search_person_by_name(person_name, top_k=50):
    """Search for images by person name from clusters"""
    # Find matching cluster
    matched_cluster = ...
    
    # Get representative face
    representative_embedding = face_index[first_face_idx]
    
    # Find similar faces
    distances = np.linalg.norm(face_index - representative_embedding, axis=1)
    top_indices = np.argsort(distances)[:top_k]
    
    # Return results with similarity scores
    return results
```

#### 3. Updated `/api/search` endpoint

```python
@app.route('/api/search', methods=['POST'])
def search():
    # Check if query matches a person name
    matched_person = check_person_name_match(query)
    
    if matched_person and search_type == 'text':
        # Use face-based search
        results = search_person_by_name(matched_person, top_k=50)
    elif search_type == 'face':
        # Direct face search
        results = search_person_fast(query, top_k=50)
    else:
        # Standard CLIP search
        results = search_images_fast(query, top_k=50)
```

---

## Frontend Updates

### Search Page (`/app/search/page.tsx`)

**New State:**
```typescript
const [matchedPerson, setMatchedPerson] = useState<string | null>(null)
const [searchType, setSearchType] = useState<string | null>(null)
```

**Response Handling:**
```typescript
const response = await axios.post("/api/search", {
  query,
  dateFrom: dateRange?.from?.toISOString(),
  dateTo: dateRange?.to?.toISOString(),
})

// Check if person name was matched
if (response.data.matched_person) {
  setMatchedPerson(response.data.matched_person)
  setSearchType(response.data.search_type)
}
```

**UI Indicator:**
When a person name is matched, shows a banner:

```tsx
{matchedPerson && searchType === 'face_by_name' && (
  <div className="mb-4 p-4 bg-primary/10 border border-primary/20 rounded-lg">
    <p className="text-sm font-medium">
      🎯 Found person: <span className="font-bold">{matchedPerson}</span>
    </p>
    <p className="text-xs text-muted-foreground mt-1">
      Showing photos using face recognition similarity
    </p>
  </div>
)}
```

---

## API Response Format

### Standard Search Response
```json
{
  "results": [...]
}
```

### Person Name Match Response
```json
{
  "results": [...],
  "search_type": "face_by_name",
  "matched_person": "John Doe"
}
```

### Individual Result with Person Match
```json
{
  "id": "0",
  "filename": "photo.jpg",
  "filepath": "/api/images/photo.jpg",
  "type": "image",
  "thumbnail": "/api/images/photo.jpg",
  "date": "2024-01-15T10:30:00",
  "score": 0.95,
  "matched_person": "John Doe"
}
```

---

## Usage Examples

### Example 1: Search for John Doe

**User Action:**
```
Search: "john doe"
```

**Backend Process:**
1. Check clusters.json → Find "John Doe" in cluster 3
2. Get representative face from John Doe's cluster
3. Search all faces by similarity
4. Return top 50 matches

**Result:**
- All photos containing John Doe (or similar faces)
- Ranked by face similarity
- Banner shows: "🎯 Found person: John Doe"

### Example 2: Partial Name Match

**User Action:**
```
Search: "sarah"
```

**Backend Process:**
1. Check clusters.json → Find "Sarah Smith"
2. Use face-based search for Sarah Smith
3. Return results

**Result:**
- Photos of Sarah Smith
- Banner shows: "🎯 Found person: Sarah Smith"

### Example 3: No Name Match

**User Action:**
```
Search: "beach sunset"
```

**Backend Process:**
1. Check clusters.json → No person match
2. Fall back to CLIP text search
3. Return semantic matches for "beach sunset"

**Result:**
- Standard CLIP search results
- No person banner shown

---

## Data Requirements

### clusters.json Structure
```json
{
  "clusters": {
    "0": {
      "cluster_id": "0",
      "name": "John Doe",
      "faces": [
        {
          "face_id": "42",
          "filename": "images/photo1.jpg",
          "bbox": [100, 150, 200, 250],
          "embedding_idx": 42
        }
      ]
    }
  },
  "noise": [...],
  "n_clusters": 15,
  "n_noise": 42
}
```

### Required Files
- `faces/clusters.json` - Person names and cluster data
- `faces/embeddings.npy` - Face embeddings array
- `faces/filenames.npy` - Image filenames array

---

## Benefits

### 1. **Better Person Search**
- More accurate than text-based CLIP search for people
- Uses actual face recognition instead of image descriptions
- Finds person even in backgrounds or group photos

### 2. **Seamless Integration**
- No UI changes needed - just type a name
- Automatic detection and switching
- Clear indicator when person match found

### 3. **Smart Matching**
- Handles partial names
- Case-insensitive
- Flexible matching (contains, partial, exact)

### 4. **High Quality Results**
- Face similarity scores
- Ranked by closeness to representative face
- No duplicate files

---

## Limitations & Considerations

### Name Conflicts
If multiple people have similar names:
- First match in iteration order is used
- Consider making names more specific

### Representative Face Quality
- Uses first face in cluster as representative
- Quality depends on cluster quality
- Consider using cluster centroid instead

### Performance
- Loads clusters.json at startup
- In-memory search is fast
- Scales with number of indexed faces

### False Positives
- Similar-looking people may appear in results
- Face similarity threshold is configurable
- Remove incorrect matches using UI

---

## Configuration

### Adjust Number of Results
In `main.py`, change `top_k`:
```python
results = search_person_by_name(matched_person, top_k=100)
```

### Adjust Name Matching Sensitivity
Modify `check_person_name_match()` logic:
```python
# Current: flexible matching
if person_name == query_lower or query_lower in person_name or person_name in query_lower:

# Strict: exact match only
if person_name == query_lower:

# Starts with
if person_name.startswith(query_lower):
```

---

## Troubleshooting

### Search doesn't detect person name
**Possible causes:**
- clusters.json not loaded
- Name spelling different
- Person not in clusters

**Solution:**
- Check console for "Loaded N people clusters"
- Verify name in `/people` page
- Check exact spelling

### Wrong person results
**Possible causes:**
- Multiple people with similar faces
- Poor quality representative face
- Need to re-cluster

**Solution:**
- Use more specific name
- Remove incorrect faces from cluster
- Re-run clustering with adjusted parameters

### Results include wrong people
**Possible causes:**
- High similarity threshold
- Similar-looking individuals
- Cluster contamination

**Solution:**
- Clean up cluster using remove face feature
- Merge if same person split across clusters
- Adjust clustering `eps` parameter

---

## Future Enhancements

### Possible Improvements
1. **Multiple Representative Faces**: Average multiple faces for better matching
2. **Confidence Threshold**: Filter out low-confidence matches
3. **Fuzzy Name Matching**: Handle typos and variations
4. **Search History**: Remember commonly searched people
5. **Auto-complete**: Suggest person names as user types
6. **Multi-person Search**: "John and Sarah" searches for both

---

## Summary

✅ **Automatic Detection**: Searches person name in clusters
✅ **Face-Based Results**: Uses face similarity instead of CLIP
✅ **Smart Matching**: Flexible name matching (exact, contains, partial)
✅ **Visual Indicator**: Shows banner when person found
✅ **Backend Connected**: Reads directly from clusters.json
✅ **High Accuracy**: Better results than text search for people
✅ **Seamless UX**: Works transparently without UI changes

**Just type a person's name and the system automatically finds their photos!** 🎯
