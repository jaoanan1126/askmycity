'use client';

import { useState, useRef, useEffect } from 'react';
import { Message } from '@/lib/types';
import { apiClient, describeError } from '@/lib/api';
import ChatMessage from '@/components/ChatMessage';
import ChatInput from '@/components/ChatInput';
import LoadingIndicator from '@/components/LoadingIndicator';
import { Building2 } from 'lucide-react';

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      role: 'assistant',
      content:
        '👋 Hi! I\'m AskMyCity, your AI assistant for San Francisco government services. Ask me anything about permits, licenses, parking, or city services!',
      timestamp: new Date(),
    },
  ]);
  const [isLoading, setIsLoading] = useState(false);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSendMessage = async (content: string) => {
    // Add user message
    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content,
      timestamp: new Date(),
    };
    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Call API
      const response = await apiClient.ask({
        question: content,
        conversation_id: conversationId || undefined,
        max_results: 3,
      });

      // Add bot response
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, botMessage]);

      // Save conversation ID
      setConversationId(response.conversation_id);
    } catch (err) {
      console.error('Error:', err);
      const errorText = describeError(err);
      setError(errorText);

      // Add error message
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `❌ ${errorText}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div 
      className="min-h-screen flex items-center justify-center p-4 bg-cover bg-center bg-fixed"
      style={{ backgroundImage: "url('/sf.png')" }}
    >
      {/* Semi-transparent overlay for better contrast */}
      <div className="absolute inset-0 bg-black/30"></div>
      
      {/* Chat container - relative positioning to appear above overlay */}
      <div className="relative bg-white/65 backdrop-blur-md rounded-3xl shadow-2xl w-full max-w-4xl h-[90vh] flex flex-col overflow-hidden">
        {/* Header */}
        <header className="bg-gradient-to-r from-primary-500/80 to-secondary-500/80 backdrop-blur-sm text-gray-800 px-6 py-5">
          <div className="flex items-center gap-3">
            <Building2 className="w-8 h-8" />
            <div>
              <h1 className="text-2xl font-bold">AskMyCity</h1>
              <p className="text-sm text-gray-800/90">
                Your AI assistant for San Francisco government services
          </p>
        </div>
          </div>
        </header>

        {/* Chat Container */}
        <div
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto bg-transparent p-6 custom-scrollbar"
        >
          {messages.map((message) => (
            <ChatMessage key={message.id} message={message} />
          ))}

          {/* Loading Indicator */}
          {isLoading && <LoadingIndicator />}
        </div>

        {/* Error Message */}
        {error && (
          <div className="px-6 py-2 bg-red-50/80 backdrop-blur-sm border-t border-red-200/50">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Input */}
        <ChatInput onSendMessage={handleSendMessage} disabled={isLoading} />
      </div>
    </div>
  );
}