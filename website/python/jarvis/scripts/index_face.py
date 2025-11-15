from jarvis.indexation.index_faces import build_face_index, cluster_faces, get_all_people, search_person, update_person_name


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python index_faces.py index                    -> build face embeddings")
        print("  python index_faces.py cluster                  -> cluster indexed faces")
        print("  python index_faces.py search 'path_to_person'  -> search for similar faces")
        print("  python index_faces.py list                     -> list all people")
        print("  python index_faces.py rename <id> <name>       -> rename a person")
        exit(1)

    cmd = sys.argv[1]

    if cmd == "index":
        build_face_index()
        print("\n💡 Next step: Run 'python index_faces.py cluster' to group faces")
    elif cmd == "cluster":
        cluster_faces()
        print("\n💡 View people: Run 'python index_faces.py list'")
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Please provide the path to the person image.")
            exit(1)
        person_image_path = sys.argv[2]
        results = search_person(person_image_path)
        print(f"Results for: '{person_image_path}'")
        for path, score in results:
            print(f"  {score:.3f}  {path}")
    elif cmd == "list":
        people = get_all_people()
        print(f"\n👥 Found {len(people)} people:\n")
        for person in people:
            print(f"  ID {person['id']}: {person['name']} ({person['face_count']} photos)")
    elif cmd == "rename":
        if len(sys.argv) < 4:
            print("Usage: python index_faces.py rename <cluster_id> <new_name>")
            exit(1)
        cluster_id = sys.argv[2]
        new_name = sys.argv[3]
        update_person_name(cluster_id, new_name)
    else:
        print("Unknown command")