import index_faces, image_index

if __name__ == "__main__":
    image_index.build_index()
    index_faces.build_face_index()
    index_faces.cluster_faces()