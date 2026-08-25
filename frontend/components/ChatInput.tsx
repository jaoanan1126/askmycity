'use client';

import { useState, FormEvent } from 'react';
import { Send } from 'lucide-react';

interface ChatInputProps {
  onSendMessage: (message: string) => void;
  disabled?: boolean;
}

export default function ChatInput({ onSendMessage, disabled = false }: ChatInputProps) {
  const [input, setInput] = useState('');

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (input.trim() && !disabled) {
      onSendMessage(input.trim());
      setInput('');
    }
  };

  return (
    <div className="border-t border-gray-200/50 bg-transparent backdrop-blur-sm p-4">
      <form onSubmit={handleSubmit} className="flex gap-3 mb-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a question about SF government services..."
          className="flex-1 px-4 py-3 border-2 border-gray-200/50 rounded-full text-[15px] bg-white/80 backdrop-blur-sm outline-none focus:border-primary-500 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={disabled}
          autoComplete="off"
        />
        <button
          type="submit"
          disabled={disabled || !input.trim()}
          className="w-12 h-12 bg-gradient-to-br from-primary-500 to-secondary-500 text-white rounded-full flex items-center justify-center hover:scale-105 active:scale-95 transition-transform disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100"
        >
          <Send className="w-5 h-5" />
        </button>
      </form>
      <p className="text-xs text-gray-500 text-center">
        💡 Try: &quot;How do I get a building permit?&quot; or &quot;What are parking meter hours?&quot;
      </p>
    </div>
  );
}