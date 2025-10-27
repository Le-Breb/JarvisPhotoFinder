import { useEffect, useState } from 'react';

interface UploadProgressProps {
  taskId: string | null;
  onComplete?: () => void;
}

export function UploadProgress({ taskId, onComplete }: UploadProgressProps) {
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState('');
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    if (!taskId) return;

    setIsVisible(true);
    setProgress(0);
    setMessage('Starting...');

    const eventSource = new EventSource(`http://localhost:5000/api/index/progress/${taskId}`);

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProgress(data.progress);
      setMessage(data.message);

      if (data.progress >= 100) {
        eventSource.close();
        setTimeout(() => {
          setIsVisible(false);
          onComplete?.();
        }, 2000);
      } else if (data.progress < 0) {
        // Error occurred
        eventSource.close();
        setTimeout(() => {
          setIsVisible(false);
        }, 5000);
      }
    };

    eventSource.onerror = () => {
      eventSource.close();
      setMessage('Connection error');
      setTimeout(() => {
        setIsVisible(false);
      }, 3000);
    };

    return () => {
      eventSource.close();
    };
  }, [taskId, onComplete]);

  if (!isVisible) return null;

  return (
    <div className="fixed bottom-4 right-4 bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4 w-80 z-50">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">
            {progress < 0 ? 'Error' : progress >= 100 ? 'Complete' : 'Indexing...'}
          </span>
          <span className="text-sm text-gray-500">{progress >= 0 ? `${progress}%` : ''}</span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <div
            className={`h-2 rounded-full transition-all duration-300 ${
              progress < 0 ? 'bg-red-500' : progress >= 100 ? 'bg-green-500' : 'bg-blue-500'
            }`}
            style={{ width: `${Math.max(0, Math.min(100, progress))}%` }}
          />
        </div>
        <p className="text-xs text-gray-600 dark:text-gray-400">{message}</p>
      </div>
    </div>
  );
}
