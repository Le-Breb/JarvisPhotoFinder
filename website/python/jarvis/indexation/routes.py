from flask import Blueprint, request, jsonify, Response
import uuid
import threading
import queue
import json
import traceback
from multiprocessing import get_context
import os

indexation_bp = Blueprint('indexation', __name__)

# Module-level variables for this blueprint
indexing_progress_queues = {}
indexing_processes = {}

def run_indexing_background(task_id, progress_queue):
    """Run indexing in a separate process with progress tracking"""
    try:
        from website.python.jarvis.indexation.indexing import send_progress
        
        print(f"🔄 Starting background indexing for task {task_id}...")
        send_progress(task_id, 0, "Starting indexing...", progress_queue)

        import website.python.jarvis.indexation.image_index as image_index

        send_progress(task_id, 10, "Indexing images...", progress_queue)
        image_index.build_index(start_percentage=10, end_percentage=50, progress_queue=progress_queue, task_id=task_id)
        send_progress(task_id, 50, "Images indexed", progress_queue)

        import website.python.jarvis.indexation.index_faces as index_faces

        send_progress(task_id, 50, "Detecting faces...", progress_queue)
        index_faces.build_face_index(start_percentage=50, end_percentage=90, progress_queue=progress_queue, task_id=task_id)

        send_progress(task_id, 90, "Clustering faces...", progress_queue)
        index_faces.cluster_faces()
        send_progress(task_id, 100, "Indexing complete", progress_queue)

        print(f"✅ Background indexing completed for task {task_id}")

    except Exception as e:
        print(f"❌ Background indexing failed: {e}")
        traceback.print_exc()
        from website.python.jarvis.indexation.indexing import send_progress
        send_progress(task_id, -1, f"Error: {str(e)}", progress_queue)

def monitor_indexing_completion(task_id, progress_queue, manager, process):
    """Monitor for indexing completion and cleanup resources"""
    from flask import current_app
    
    try:
        while True:
            try:
                progress_data = progress_queue.get(timeout=1)

                if progress_data.get('progress') == 100:
                    print(f"🔄 Reloading resources after indexing completion")
                    with current_app.app_context():
                        from jarvis.main import load_clip_resources, load_face_resources
                        load_clip_resources()
                        load_face_resources()
                    print(f"✅ Resources reloaded")
                    break
                elif progress_data.get('progress', 0) < 0:
                    print(f"❌ Indexing failed for task {task_id}")
                    break

            except queue.Empty:
                if not process.is_alive():
                    print(f"⚠️ Process died unexpectedly for task {task_id}")
                    break
                continue

    except Exception as e:
        print(f"❌ Error in completion monitor: {e}")
        traceback.print_exc()
    finally:
        cleanup_indexing_task(task_id, process, manager, progress_queue)

def cleanup_indexing_task(task_id, process, manager, progress_queue):
    """Clean up all resources associated with an indexing task"""
    try:
        print(f"🧹 Cleaning up indexing task {task_id}")

        if process.is_alive():
            process.join(timeout=5)
            if process.is_alive():
                print(f"⚠️ Force terminating process {process.pid}")
                process.terminate()
                process.join(timeout=2)

        try:
            while not progress_queue.empty():
                progress_queue.get_nowait()
        except:
            pass

        try:
            manager.shutdown()
        except:
            pass

        try:
            process.close()
        except:
            pass

        if task_id in indexing_progress_queues:
            del indexing_progress_queues[task_id]
        if task_id in indexing_processes:
            del indexing_processes[task_id]

        import gc
        gc.collect()

        print(f"✅ Cleaned up task {task_id}")

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        traceback.print_exc()

@indexation_bp.route('/api/index/trigger', methods=['POST'])
def trigger_indexing():
    """Trigger indexing in separate process and return task ID"""
    try:
        task_id = str(uuid.uuid4())
        ctx = get_context('spawn')
        manager = ctx.Manager()
        progress_queue = manager.Queue()
        
        indexing_progress_queues[task_id] = {
            'queue': progress_queue,
            'manager': manager
        }

        proc = ctx.Process(
            target=run_indexing_background,
            args=(task_id, progress_queue),
            daemon=False
        )
        proc.start()

        indexing_processes[task_id] = proc

        monitor_thread = threading.Thread(
            target=monitor_indexing_completion,
            args=(task_id, progress_queue, manager, proc),
            daemon=True
        )
        monitor_thread.start()

        return jsonify({
            'status': 'started',
            'message': 'Indexing started in background process',
            'task_id': task_id,
            'pid': proc.pid
        }), 202

    except Exception as e:
        print(f"❌ Error triggering indexing: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@indexation_bp.route('/api/index/progress/<task_id>')
def indexing_progress(task_id):
    """SSE endpoint for indexing progress"""
    def generate():
        if task_id not in indexing_progress_queues:
            yield f"data: {json.dumps({'progress': -1, 'message': 'Task not found'})}\n\n"
            return

        progress_queue = indexing_progress_queues[task_id]['queue']

        while True:
            try:
                progress_data = progress_queue.get(timeout=30)
                yield f"data: {json.dumps(progress_data)}\n\n"

                if progress_data['progress'] >= 100 or progress_data['progress'] < 0:
                    break

            except queue.Empty:
                yield f": keepalive\n\n"
            except:
                break

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )