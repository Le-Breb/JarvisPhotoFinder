# Bidirectional Search Implementation

## Overview
Implemented a bidirectional rescoring mechanism that combines face recognition and CLIP semantic search for better results when searching for "person + context" queries (e.g., "Alice at the beach").

## How It Works

### Previous Approach (Simple Combination)
1. Get top 50 face matches for "Alice"
2. Get top 50 CLIP matches for "beach"
3. Combine scores using multiplication

**Problem**: Many irrelevant results from each model that never get scored by the other model.

### New Approach (Bidirectional Rescoring)
When you search for "Alice at the beach":

1. **Face → CLIP Rescoring**:
   - Get top 100 face matches for "Alice"
   - Re-score ALL those images using CLIP with context "at the beach"
   - Now each face result has BOTH face score AND CLIP score

2. **CLIP → Face Rescoring**:
   - Get top 100 CLIP matches for "at the beach"
   - Re-score ALL those images using face recognition for "Alice"
   - Now each CLIP result has BOTH CLIP score AND face score

3. **Combination**:
   - Merge both sets of results
   - Use multiplicative scoring: `(face_similarity + threshold) × (clip_score + threshold)`
   - Sort by combined score

## Key Functions

### `rescore_with_clip(face_results, context_query, top_k=50)`
- Takes face search results
- Extracts CLIP embeddings for each image
- Calculates similarity with context query
- Returns results with added `clip_score` field

### `rescore_with_face(text_results, person_name, top_k=50)`
- Takes CLIP search results
- Finds person's face cluster and medoid
- Calculates face distance for each image
- Returns results with added `face_score` field

### `combine_search_results(face_results, text_results, threshold=0.01)`
- Updated to handle rescored results
- Detects if results have `clip_score` or `face_score` from rescoring
- Uses bidirectional scores for better ranking
- Handles three cases:
  - Face→CLIP: Face results rescored with CLIP (⭐⭐)
  - CLIP→Face: CLIP results rescored with face (⭐⭐)
  - Single model only: Results from only one model

## Benefits

1. **Better Precision**: Top face matches are validated against context
2. **Better Recall**: Top context matches are validated for person presence
3. **Balanced Results**: Images good in BOTH metrics rise to the top
4. **Fewer False Positives**: Images of wrong person in right context (or vice versa) get lower scores

## Example Output

```
🔄 Re-scoring 100 face results with CLIP context: 'at the beach'
🔄 Re-scoring 100 CLIP results with face recognition for: 'Alice'
📊 Face→CLIP: image1.jpg | Face: 0.891, CLIP: 0.742 = 0.749 ⭐⭐
📊 CLIP→Face: image2.jpg | Face: 0.654, CLIP: 0.823 = 0.597 ⭐⭐
📊 BOTH: image3.jpg | Face: 0.912, CLIP: 0.891 = 0.902 ⭐
```

## Configuration

- `top_k=100`: Number of initial results to fetch before rescoring
- `threshold=0.01`: Minimum score added to prevent zero multiplication
- Both values can be tuned for performance vs. quality trade-off

## Performance Considerations

- Each rescoring pass requires computing embeddings/distances
- Uses top 100 from each model (increased from 50) to capture more candidates
- Final results are still limited to user's requested top_k (usually 50)
- CLIP rescoring uses pre-computed image embeddings (fast)
- Face rescoring uses pre-computed face embeddings (fast)
