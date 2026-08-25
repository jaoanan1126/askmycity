import { Message } from '@/lib/types';
import { Bot, User } from 'lucide-react';
import SourcesList from './SourcesList';

interface ChatMessageProps {
  message: Message;
}

export default function ChatMessage({ message }: ChatMessageProps) {
  const isBot = message.role === 'assistant';

  return (
    <div
      className={`flex gap-3 mb-1 animate-fade-in ${
        isBot ? '' : 'flex-row-reverse'
      }`}
    >
      {/* Avatar */}
      <div
        className={`w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 ${
          isBot
            ? 'bg-gradient-to-br from-primary-500 to-secondary-500'
            : 'bg-gradient-to-br from-pink-500 to-rose-500'
        }`}
      >
        {isBot ? (
          <Bot className="w-5 h-5 text-gray-800" />
        ) : (
          <User className="w-5 h-5 text-gray-800" />
        )}
      </div>

      {/* Message Content */}
      <div className={`flex-1 max-w-[70%] ${isBot ? '' : 'flex flex-col items-end'}`}>
        <div
          className={`rounded-2xl px-4 py-3 shadow-sm ${
            isBot
              ? 'bg-white text-gray-800'
              : 'bg-white text-gray-800 from-primary-500 to-secondary-500 text-gray'
          }`}
        >
          <p className="text-[15px] leading-relaxed whitespace-pre-wrap">
            {message.content}
          </p>
        </div>

        {/* Sources */}
        {message.sources && message.sources.length > 0 && (
          <SourcesList sources={message.sources} />
        )}
      </div>
    </div>
  );
}