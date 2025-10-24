# Advanced People Management Features

## New Features Added

### ✅ Merge People Clusters
Combine multiple people into one when the same person is split across different clusters.

### ✅ Delete Entire Person
Remove a person cluster and move all their photos to unclustered faces.

### ✅ All Backend-Connected
Every operation directly updates `clusters.json` - no manual file editing needed!

---

## 1. Merge People

### How It Works

**UI Flow:**
1. Navigate to `/people`
2. Click **"Merge People"** button (top right)
3. Selection mode activates - checkboxes appear on cards
4. Click person cards to select (minimum 2)
5. Selected cards show blue ring border
6. Click **"Merge (N)"** button
7. Enter name for merged person
8. Confirm merge

**What Happens:**
- All faces from selected clusters combined
- Original clusters deleted from backend
- New merged cluster created with first cluster's ID
- Changes saved to `clusters.json` immediately

**API:** `POST /api/people/merge`
```json
{
  "clusterIds": ["0", "3", "7"],
  "name": "John Doe"
}
```

---

## 2. Delete Person

### How It Works

**UI Flow:**
1. Navigate to `/people/[id]` (person detail page)
2. Click **"Delete Person"** button (top right, red)
3. Confirm in dialog showing:
   - Person name
   - Number of photos affected
   - Warning about moving to unclustered
4. Click "Delete Person" to confirm

**What Happens:**
- Person cluster removed from backend
- All faces moved to "noise" (unclustered faces)
- Cluster and noise counts updated
- Changes saved to `clusters.json` immediately
- User redirected to `/people`

**API:** `DELETE /api/people/[id]`

---

## 3. Remove Individual Face

### How It Works

**UI Flow:**
1. Navigate to `/people/[id]` (person detail page)
2. Hover over photo with incorrect person
3. Trash icon appears (top right of photo)
4. Click trash icon (stops event, won't open viewer)
5. Confirm in dialog
6. Face removed

**What Happens:**
- Face removed from cluster
- If cluster has <2 faces remaining:
  - Entire cluster deleted
  - Remaining faces moved to noise
  - User redirected to `/people`
- Changes saved to `clusters.json` immediately

**API:** `DELETE /api/people/[id]/faces/[faceId]`

---

## Backend Implementation

### API Routes Created

1. **`/app/api/people/merge/route.ts`**
   - POST endpoint for merging clusters
   - Validates minimum 2 clusters
   - Combines all faces
   - Updates `clusters.json`

2. **`/app/api/people/[id]/route.ts`** (updated)
   - Added DELETE method
   - Removes cluster
   - Moves faces to noise
   - Updates counts

3. **`/app/api/people/[id]/faces/[faceId]/route.ts`** (existing)
   - DELETE method for removing individual faces
   - Handles cluster cleanup

### Data Flow

```
User Action (UI)
    ↓
API Request (axios)
    ↓
API Route Handler (Next.js)
    ↓
Read clusters.json
    ↓
Modify Data
    ↓
Write clusters.json
    ↓
Response to UI
    ↓
UI Update / Redirect
```

### File Updates

All operations directly modify:
```
website/python/faces/clusters.json
```

**Structure:**
```json
{
  "clusters": {
    "0": {
      "cluster_id": "0",
      "name": "Person Name",
      "faces": [...]
    }
  },
  "noise": [...],
  "n_clusters": 15,
  "n_noise": 42
}
```

---

## UI Components

### People Page (`/app/people/page.tsx`)

**New State:**
- `selectionMode` - Toggle selection/normal mode
- `selectedPeople` - Set of selected cluster IDs
- `showMergeDialog` - Show/hide merge dialog
- `mergeName` - Name for merged person
- `merging` - Loading state during merge

**New UI Elements:**
- "Merge People" button (normal mode)
- "Cancel" and "Merge (N)" buttons (selection mode)
- Checkboxes on person cards (selection mode)
- Blue ring border on selected cards
- Merge dialog with name input and selected people list

### Person Detail Page (`/app/people/[id]/page.tsx`)

**New State:**
- `showDeleteDialog` - Show/hide delete dialog
- `deleting` - Loading state during deletion

**New UI Elements:**
- "Delete Person" button (red, top right)
- Delete confirmation dialog
- Shows person name and photo count in dialog

---

## Usage Examples

### Merge Example
```
Scenario: Same person split into 3 clusters

1. /people shows:
   - Person 0 (5 photos)
   - Person 3 (8 photos)  
   - Person 7 (12 photos)
   All are the same person!

2. Click "Merge People"
3. Select all 3 people
4. Click "Merge (3)"
5. Enter "John Doe"
6. Click "Merge People"

Result:
- One "John Doe" with 25 photos
- Clusters 3 and 7 deleted
- Cluster 0 becomes merged cluster
```

### Delete Example
```
Scenario: Incorrect cluster needs removal

1. /people/5 shows "Unknown Person 5"
2. All 3 photos are false positives
3. Click "Delete Person"
4. Confirm deletion

Result:
- Cluster 5 removed
- 3 faces moved to noise
- Can re-cluster later
```

### Remove Face Example
```
Scenario: 1 wrong photo in cluster

1. /people/2 shows "Sarah" with 10 photos
2. Photo #7 is actually Mike
3. Hover over photo #7
4. Click trash icon
5. Confirm removal

Result:
- "Sarah" now has 9 photos (correct ones)
- Face #7 moved to noise
```

---

## Best Practices

### ✅ Do's
- Merge when same person split across clusters
- Delete when entire cluster is incorrect
- Remove faces for individual mistakes
- Confirm selections before merging
- Review merge dialog list before confirming

### ❌ Don'ts
- Don't merge different people
- Don't delete when you just need to remove a few faces
- Don't merge without reviewing selected people
- Don't re-cluster immediately after manual fixes (will lose changes)

---

## Security

All endpoints:
- ✅ Require authentication (NextAuth session)
- ✅ Return 401 if unauthorized
- ✅ Validate input data
- ✅ Handle errors gracefully

---

## Testing Checklist

- [ ] Merge 2 people - verify faces combined
- [ ] Merge 3+ people - verify all faces combined
- [ ] Merge with custom name - verify name persists
- [ ] Delete person from detail page - verify cluster removed
- [ ] Delete person - verify redirected to /people
- [ ] Remove face - verify face removed from cluster
- [ ] Remove face (cluster <2) - verify cluster deleted
- [ ] Selection mode - verify checkboxes appear
- [ ] Selection mode - verify cancel works
- [ ] Merge dialog - verify selected people listed
- [ ] All operations - verify `clusters.json` updated

---

## Files Modified

### Created:
- `/app/api/people/merge/route.ts`

### Modified:
- `/app/api/people/[id]/route.ts` - Added DELETE method
- `/app/people/page.tsx` - Added selection mode and merge UI
- `/app/people/[id]/page.tsx` - Added delete person button

---

## Summary

🎉 **All requested features implemented:**

✅ **Merge people clusters** - Select multiple, combine into one
✅ **Delete entire person** - Remove cluster, move to noise  
✅ **Backend-connected** - All operations update `clusters.json` directly
✅ **Confirmation dialogs** - Prevent accidental changes
✅ **Clean UI** - Selection mode, checkboxes, clear feedback
✅ **Secure** - Authentication required for all operations

**Everything modifies the backend directly** - no manual file editing needed!
