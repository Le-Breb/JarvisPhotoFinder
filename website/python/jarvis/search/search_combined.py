import torch
import clip
import numpy as np


def combine_search_results(face_results, text_results, threshold_faces=0.0, threshold_text=0.1):
    """Combine face and text search results using multiplicative scoring with threshold"""
    combined = {}

    for result in face_results:
        filepath = result['filepath']
        face_score = float(result.get('face_score_original', result['score']))
        clip_score = float(result.get('clip_score', 0))

        face_similarity = 1.0 / (1.0 + abs(face_score))

        if clip_score > 0:
            combined_score = float((face_similarity + threshold_faces) * (clip_score + threshold_text))
            combined[filepath] = {
                'result': result,
                'face_score': face_score,
                'face_similarity': face_similarity,
                'text_score': clip_score,
                'combined_score': combined_score
            }
            print(f"📊 Face→CLIP: {filepath.split('/')[-1]} | Face: {face_similarity:.3f}, CLIP: {clip_score:.3f} = {combined_score:.3f} ⭐⭐")
        else:
            combined_score = float((face_similarity + threshold_faces) * threshold_text)
            combined[filepath] = {
                'result': result,
                'face_score': face_score,
                'face_similarity': face_similarity,
                'text_score': 0.0,
                'combined_score': combined_score
            }
            print(f"📊 Face only: {filepath.split('/')[-1]} | Face: {face_similarity:.6f} = {combined_score:.6f}")
    
    for result in text_results:
        filepath = result['filepath']
        text_score = float(result.get('clip_score_original', result['score']))
        face_score = float(result.get('face_score', 0))
        
        if filepath in combined:
            existing = combined[filepath]
            if face_score != 0:
                face_similarity = 1.0 / (1.0 + abs(face_score))
                combined_score = float((face_similarity + threshold_faces) * (text_score + threshold_text))
                combined[filepath]['face_score'] = face_score
                combined[filepath]['face_similarity'] = face_similarity
                combined[filepath]['text_score'] = text_score
                combined[filepath]['combined_score'] = combined_score
                combined[filepath]['result']['score'] = combined_score
                print(f"📊 CLIP→Face: {filepath.split('/')[-1]} | Face: {face_similarity:.3f}, CLIP: {text_score:.3f} = {combined_score:.3f} ⭐⭐")
        else:
            if face_score != 0:
                face_similarity = 1.0 / (1.0 + abs(face_score))
                combined_score = float((face_similarity + threshold_faces) * (text_score + threshold_text))
                combined[filepath] = {
                    'result': result,
                    'face_score': face_score,
                    'face_similarity': face_similarity,
                    'text_score': text_score,
                    'combined_score': combined_score
                }
                combined[filepath]['result']['score'] = combined_score
                print(f"📊 CLIP→Face: {filepath.split('/')[-1]} | Face: {face_similarity:.3f}, CLIP: {text_score:.3f} = {combined_score:.3f} ⭐")
            else:
                combined_score = float(threshold_faces * (text_score + threshold_text))
                combined[filepath] = {
                    'result': result,
                    'face_score': 0.0,
                    'face_similarity': 0.0,
                    'text_score': text_score,
                    'combined_score': combined_score
                }
                combined[filepath]['result']['score'] = combined_score
                print(f"📊 Text only: {filepath.split('/')[-1]} | CLIP: {text_score:.3f} = {combined_score:.3f}")
    
    sorted_results = sorted(combined.values(), key=lambda x: x['combined_score'], reverse=True)
    print(f"\n🏆 Top 5 results after combining:")
    for i, item in enumerate(sorted_results[:5]):
        print(f"  {i + 1}. {item['result']['filepath'].split('/')[-1]} | Score: {item['combined_score']:.3f}")

    return [item['result'] for item in sorted_results]