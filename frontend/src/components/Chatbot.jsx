import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, X, Bot, User, RefreshCcw, Send, Loader } from 'lucide-react';
import axios from 'axios'; // Import axios to make API calls

// This is the updated function to call YOUR backend
async function getGeminiResponse(history, prompt) {
  try {
    // We're calling '/api/chat' which Vite will proxy to http://localhost:5001/api/chat
    const response = await axios.post('/api/chat', {
      history: history,
      prompt: prompt,
    });
    return response.data.text;
  } catch (error) {
    console.error('Error fetching from backend API:', error);
    return 'Sorry, I ran into an error connecting to the server. Please try again.';
  }
}

const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: 'model',
      text: "Hi! I'm Ascend AI. How can I help you with your fitness and nutrition goals today?",
    },
  ]);
  // This history is just for the API, not the UI
  const [chatHistory, setChatHistory] = useState([]);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(scrollToBottom, [messages]);

  const clearChat = () => {
    setMessages([
      {
        role: 'model',
        text: "Hi! I'm Ascend AI. How can I help you with your fitness and nutrition goals today?",
      },
    ]);
    setChatHistory([]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const input = e.target.elements.chatInput;
    const prompt = input.value.trim();
    if (!prompt || isLoading) return;

    // Add user message to UI
    setMessages((prev) => [...prev, { role: 'user', text: prompt }]);
    setIsLoading(true);
    input.value = '';

    try {
      const responseText = await getGeminiResponse(chatHistory, prompt);
      
      // Add model response to UI
      setMessages((prev) => [...prev, { role: 'model', text: responseText }]);
      
      // Update history for next call
      setChatHistory(prev => [
          ...prev,
          { role: "user", parts: [{ text: prompt }] },
          { role: "model", parts: [{ text: responseText }] }
      ]);

    } catch (error) {
      console.error('Error getting Gemini response:', error);
      setMessages((prev) => [
        ...prev,
        { role: 'model', text: 'Sorry, I ran into an error. Please try again.' },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {/* Chat Bubble */}
      <div
        className={`fixed bottom-8 right-8 bg-accent text-gray-900 w-16 h-16 rounded-full flex items-center justify-center cursor-pointer shadow-2xl z-50 transition-all duration-300 ${
          isOpen ? 'scale-0 opacity-0' : 'scale-100 opacity-100'
        } hover:scale-110`}
        onClick={() => setIsOpen(true)}
      >
        <MessageSquare size={32} />
      </div>

      {/* Modal Overlay */}
      <div
        className={`fixed inset-0 bg-black/70 z-50 transition-opacity duration-300 ${
          isOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
        onClick={() => setIsOpen(false)}
      ></div>
      
      {/* Chat Sidebar */}
      <div
        className={`fixed top-0 right-0 h-full w-full max-w-lg bg-gray-900 border-l border-gray-700 shadow-2xl z-50 flex flex-col transition-transform duration-300 ease-in-out ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700 flex-shrink-0">
          <h3 className="text-lg font-bold text-white">
            Ask <span className="text-accent">Ascend AI</span>
          </h3>
          <div className="flex items-center gap-2">
            <button
              className="text-gray-400 hover:text-white transition-colors"
              onClick={clearChat}
              title="Clear Chat"
            >
              <RefreshCcw size={20} />
            </button>
            <button
              className="text-gray-400 hover:text-white transition-colors"
              onClick={() => setIsOpen(false)}
              title="Close Chat"
            >
              <X size={24} />
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 p-4 space-y-6 overflow-y-auto">
          {messages.map((msg, index) => (
            <ChatMessage key={index} role={msg.role} text={msg.text} />
          ))}
          {isLoading && <TypingIndicator />}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="p-4 border-t border-gray-700 bg-gray-900 flex-shrink-0">
          <form className="flex gap-2" onSubmit={handleSubmit}>
            <input
              type="text"
              name="chatInput"
              className="flex-1 bg-gray-800 border border-gray-600 rounded-lg p-3 text-white focus:border-accent focus:ring-accent focus:ring-1 outline-none"
              placeholder="Type your message..."
              autoComplete="off"
              disabled={isLoading}
            />
            <button
              type="submit"
              className="bg-accent text-gray-900 p-3 rounded-lg font-bold flex items-center justify-center w-14 h-14 flex-shrink-0 disabled:opacity-50"
              disabled={isLoading}
            >
              {isLoading ? <Loader className="animate-spin loader-dark" /> : <Send size={20} />}
            </button>
          </form>
        </div>
      </div>
    </>
  );
};

const ChatMessage = ({ role, text }) => {
  const isUser = role === 'user';
  return (
    <div className={`flex items-start gap-3 ${isUser ? 'flex-row-reverse' : ''}`}>
      <div
        className={`w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center ${
          isUser ? 'bg-gray-700 text-gray-300' : 'bg-accent/20 text-accent'
        }`}
      >
        {isUser ? <User size={20} /> : <Bot size={20} />}
      </div>
      <div
        className={`p-3 max-w-xs rounded-lg whitespace-pre-wrap ${
          isUser
            ? 'bg-gray-800 text-white rounded-br-none'
            : 'bg-accent/10 text-gray-300 rounded-bl-none'
        }`}
      >
        {text}
      </div>
    </div>
  );
};

const TypingIndicator = () => (
  <div className="flex items-start gap-3">
    <div className="w-8 h-8 flex-shrink-0 rounded-full flex items-center justify-center bg-accent/20 text-accent">
      <Bot size={20} />
    </div>
    <div className="p-3 text-gray-400">
      Ascend AI is typing
      <span className="animate-pulse delay-100">.</span>
      <span className="animate-pulse delay-200">.</span>
      <span className="animate-pulse delay-300">.</span>
    </div>
  </div>
);

export default Chatbot;