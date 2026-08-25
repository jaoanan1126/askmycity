export default function LoadingIndicator() {
    return (
      <div className="flex items-center gap-3 px-6 py-4 text-gray-600">
        <div className="flex gap-1">
          <span className="w-2 h-2 bg-primary-500 rounded-full animate-bounce-dot"></span>
          <span
            className="w-2 h-2 bg-primary-500 rounded-full animate-bounce-dot"
            style={{ animationDelay: '0.16s' }}
          ></span>
          <span
            className="w-2 h-2 bg-primary-500 rounded-full animate-bounce-dot"
            style={{ animationDelay: '0.32s' }}
          ></span>
        </div>
        <span className="text-sm">Thinking...</span>
      </div>
    );
  }