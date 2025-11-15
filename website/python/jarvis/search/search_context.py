import torch
import clip
import numpy as np
from jarvis.config import DEFAULT_SEARCH_COUNT_CONTEXT
from flask import current_app


def search_images_fast(query, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), top_k=DEFAULT_SEARCH_COUNT_CONTEXT):
    """Fast search using pre-loaded resources"""
    clip_model = current_app.config['CLIP_MODEL']
    clip_index = current_app.config['CLIP_INDEX']
    clip_filenames = current_app.config['CLIP_FILENAMES']

    with torch.no_grad():
        text = clip.tokenize([query]).to(device)
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    query_vec = text_features.cpu().numpy().astype(np.float32)
    scores, indices = clip_index.search(query_vec, top_k)

    results = []
    for rank, i in enumerate(indices[0]):
        filename = clip_filenames[i]
        if filename.startswith('images/'):
            filename = filename[7:]

        image_path = f'/api/images/{filename}'
        results.append({
            'filepath': image_path,
            'thumbnail': image_path,
            'score': float(scores[0][rank])
        })

    return results

def rescore_with_clip(face_results, context_query, clip_model, clip_index, clip_filenames, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """Re-score face search results using CLIP for context matching"""
    if not context_query or not face_results:
        return face_results

    print(f"🔄 Re-scoring {len(face_results)} face results with CLIP context: '{context_query}'")

    filenames_to_check = []
    filepath_to_face_result = {}
    for result in face_results:
        filepath = result['filepath']
        filename = filepath.split('/')[-1]
        for idx, clip_fname in enumerate(clip_filenames):
            clip_fname_clean = clip_fname[7:] if clip_fname.startswith('images/') else clip_fname
            if clip_fname_clean == filename:
                filenames_to_check.append((idx, filepath))
                filepath_to_face_result[filepath] = result
                break

    if not filenames_to_check:
        return face_results

    with torch.no_grad():
        text = clip.tokenize([context_query]).to(device)
        text_features = clip_model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_vec = text_features.cpu().numpy().astype(np.float32)

    rescored_results = []
    for clip_idx, filepath in filenames_to_check:
        image_embedding = np.array([clip_index.reconstruct(int(clip_idx))], dtype=np.float32)
        image_embedding = image_embedding / np.linalg.norm(image_embedding)
        clip_score = float(np.dot(text_vec, image_embedding.T)[0][0])

        face_result = filepath_to_face_result[filepath]
        rescored_results.append({
            **face_result,
            'clip_score': clip_score,
            'face_score_original': float(face_result['score'])
        })

    return rescored_results

