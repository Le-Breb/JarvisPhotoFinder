def send_progress(task_id, percent, message, progress_queue):
    """Send progress update via the provided queue"""
    try:
        progress_queue.put({
            'task_id': task_id,
            'progress':  round(percent),
            'message': message
        })
    except Exception as e:
        print(f"Failed to send progress: {e}")