"use client";

import React, { useState, useEffect, useRef } from "react";
import { MessageSquare, FileText, Upload, Trash2, Menu, Plus, Send, X, FileUp } from "lucide-react";
import ReactMarkdown from "react-markdown";

// Types
type Role = "user" | "assistant";
type Mode = "general" | "document";

interface Message {
  role: Role;
  content: string;
  sources?: any[];
}

interface Conversation {
  id: string;
  title: string;
  mode: Mode;
  updated_at: string;
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("general");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [availableFiles, setAvailableFiles] = useState<any[]>([]);
  const [activeConvId, setActiveConvId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Fetch history on load
  useEffect(() => {
    fetchHistory();
    fetchFiles();
  }, []);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const fetchFiles = async () => {
    try {
      const res = await fetch(`http://${window.location.hostname}:8000/api/status`);
      const data = await res.json();
      if (data && data.documents) {
        setAvailableFiles(data.documents);
      }
    } catch (e) {
      console.error("Failed to fetch files", e);
    }
  };

  const fetchHistory = async () => {
    try {
      const res = await fetch(`http://${window.location.hostname}:8000/api/history`);
      const data = await res.json();
      setConversations(data);
    } catch (e) {
      console.error("Failed to fetch history", e);
    }
  };

  const loadConversation = async (id: string) => {
    try {
      const res = await fetch(`http://${window.location.hostname}:8000/api/conversations/${id}`);
      const data = await res.json();
      setActiveConvId(data.id);
      setMode(data.mode);
      
      const parsedMessages = data.messages.map((m: any) => {
        if (m.role === "assistant") {
          try {
            const parsed = JSON.parse(m.content);
            return { role: m.role, content: parsed.text || parsed.content, sources: parsed.sources };
          } catch {
            return { role: m.role, content: m.content };
          }
        }
        return m;
      });
      setMessages(parsedMessages);
    } catch (e) {
      console.error("Failed to load conversation", e);
    }
  };

  const createNewChat = () => {
    setActiveConvId(null);
    setMessages([]);
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    
    const file = e.target.files[0];
    const formData = new FormData();
    formData.append("file", file);
    
    setIsUploading(true);
    setUploadStatus("Uploading & Indexing...");
    
    try {
      const res = await fetch(`http://${window.location.hostname}:8000/api/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setUploadStatus(`Success: ${data.chunks} chunks indexed.`);
        setMode("document");
        fetchFiles(); // Refresh file list
      } else {
        setUploadStatus(`Error: ${data.detail}`);
      }
    } catch (e) {
      setUploadStatus("Failed to upload file.");
    } finally {
      setIsUploading(false);
      setTimeout(() => setUploadStatus(null), 5000);
    }
  };

  const sendMessage = async (e?: React.FormEvent) => {
    e?.preventDefault();
    if (!input.trim() || isStreaming) return;

    const userMessage = input.trim();
    setInput("");
    
    const newMessages = [...messages, { role: "user" as const, content: userMessage }];
    setMessages(newMessages);
    setIsStreaming(true);

    try {
      const convId = activeConvId || crypto.randomUUID();
      if (!activeConvId) setActiveConvId(convId);

      const res = await fetch(`http://${window.location.hostname}:8000/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: userMessage,
          mode,
          conversation_id: convId,
        }),
      });

      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let assistantMsg = "";
      let sources: any[] = [];
      
      setMessages([...newMessages, { role: "assistant", content: "", sources: [] }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n\n");
        
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.replace("data: ", "");
            if (dataStr === "[DONE]") {
              fetchHistory();
              break;
            }
            try {
              const data = JSON.parse(dataStr);
              if (data.type === "token") {
                assistantMsg += data.content;
              } else if (data.type === "sources") {
                sources = data.content;
              } else if (data.type === "error") {
                assistantMsg += `\n\n**Error:** ${data.content}`;
              }
              
              setMessages([
                ...newMessages,
                { role: "assistant", content: assistantMsg, sources }
              ]);
            } catch (e) {
              // Ignore incomplete JSON parses in stream
            }
          }
        }
      }
    } catch (error) {
      console.error("Chat error:", error);
    } finally {
      setIsStreaming(false);
    }
  };

  return (
    <div suppressHydrationWarning className="flex h-screen bg-gray-900 text-white font-sans overflow-hidden">
      {/* Sidebar */}
      <div suppressHydrationWarning className={`fixed md:relative z-20 flex-shrink-0 h-full bg-gray-950 border-r border-gray-800 transition-all duration-300 ${isSidebarOpen ? "w-64" : "w-0"} overflow-hidden`}>
        <div className="flex flex-col h-full p-3 w-64">
          <button 
            onClick={createNewChat}
            className="flex items-center gap-3 p-3 w-full rounded-md border border-gray-700 hover:bg-gray-800 transition-colors"
          >
            <Plus size={18} suppressHydrationWarning />
            <span className="text-sm font-medium">New Chat</span>
          </button>
          
          <div className="mt-6 flex-1 overflow-y-auto">
            <h3 className="text-xs text-gray-500 font-semibold mb-3 px-3 uppercase tracking-wider">Recent</h3>
            {conversations.map((conv) => (
              <button
                key={conv.id}
                onClick={() => loadConversation(conv.id)}
                className={`flex items-center gap-3 w-full p-3 rounded-md text-sm text-left truncate transition-colors ${activeConvId === conv.id ? "bg-gray-800 text-white" : "text-gray-300 hover:bg-gray-800/50"}`}
              >
                {conv.mode === "general" ? <MessageSquare size={16} suppressHydrationWarning /> : <FileText size={16} suppressHydrationWarning />}
                <span className="truncate">{conv.title}</span>
              </button>
            ))}

            {availableFiles.length > 0 && (
              <div className="mt-8">
                <h3 className="text-xs text-gray-500 font-semibold mb-3 px-3 uppercase tracking-wider">Available Files</h3>
                {availableFiles.map((file: any, idx) => (
                  <div key={idx} className="flex items-center gap-3 w-full p-3 rounded-md text-sm text-left truncate text-gray-400">
                    <FileText size={16} suppressHydrationWarning />
                    <span className="truncate" title={file.source_file}>{file.source_file}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-full min-w-0 bg-gray-900">
        {/* Header */}
        <header className="h-14 border-b border-gray-800 flex items-center justify-between px-4 sticky top-0 bg-gray-900/80 backdrop-blur-md z-10">
          <div className="flex items-center gap-3">
            <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 hover:bg-gray-800 rounded-md text-gray-400 hover:text-white">
              <Menu size={20} suppressHydrationWarning />
            </button>
            
            {/* Mode Toggle */}
            <div className="flex bg-gray-950 rounded-lg p-1 border border-gray-800">
              <button
                onClick={() => setMode("general")}
                className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center gap-2 transition-colors ${mode === "general" ? "bg-gray-800 text-white" : "text-gray-400 hover:text-white"}`}
              >
                <MessageSquare size={16} suppressHydrationWarning /> <span className="hidden sm:inline">General AI</span>
              </button>
              <button
                onClick={() => setMode("document")}
                className={`px-3 py-1.5 rounded-md text-sm font-medium flex items-center gap-2 transition-colors ${mode === "document" ? "bg-blue-600/20 text-blue-400" : "text-gray-400 hover:text-white"}`}
              >
                <FileText size={16} suppressHydrationWarning /> <span className="hidden sm:inline">Ask Documents</span>
              </button>
            </div>
          </div>

          {/* Upload Button */}
          <div className="relative">
            <input 
              type="file" 
              id="file-upload" 
              className="hidden" 
              onChange={handleFileUpload}
              disabled={isUploading}
            />
            <label 
              htmlFor="file-upload" 
              className="flex items-center gap-2 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-md text-sm font-medium cursor-pointer transition-colors"
            >
              <FileUp size={16} suppressHydrationWarning />
              <span className="hidden sm:inline">{isUploading ? "Uploading..." : "Upload File"}</span>
            </label>
            {uploadStatus && (
              <div className="absolute right-0 top-10 mt-2 w-48 p-2 text-xs bg-gray-800 border border-gray-700 rounded-md shadow-lg z-50 text-center">
                {uploadStatus}
              </div>
            )}
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto p-4 md:p-8">
          <div className="max-w-3xl mx-auto flex flex-col gap-8 pb-32">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full mt-32 text-center text-gray-400">
                <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mb-6">
                  {mode === "general" ? <MessageSquare size={32} suppressHydrationWarning /> : <FileText size={32} className="text-blue-400" suppressHydrationWarning />}
                </div>
                <h2 className="text-2xl font-bold text-white mb-2">
                  {mode === "general" ? "How can I help you today?" : "Ask questions about your documents."}
                </h2>
                <p className="max-w-md">
                  {mode === "general" 
                    ? "Ask me anything, I am a helpful AI assistant." 
                    : "Upload a PDF, TXT, CSV, or Markdown file to get grounded answers with source citations."}
                </p>
              </div>
            ) : (
              messages.map((msg, i) => (
                <div key={i} className={`flex gap-4 ${msg.role === "assistant" ? "bg-transparent" : "bg-transparent"}`}>
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${msg.role === "user" ? "bg-blue-600" : "bg-emerald-600"}`}>
                    {msg.role === "user" ? "U" : "AI"}
                  </div>
                  <div className="flex-1 overflow-hidden">
                    <div className="text-sm font-semibold mb-1 text-gray-300">
                      {msg.role === "user" ? "You" : "Assistant"}
                    </div>
                    <div className="prose prose-invert prose-p:leading-relaxed prose-pre:bg-gray-950 prose-pre:border prose-pre:border-gray-800 max-w-none">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                    
                    {/* Source Citations */}
                    {msg.sources && msg.sources.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-gray-800">
                        <h4 className="text-xs uppercase tracking-wider text-gray-500 font-semibold mb-2">Sources</h4>
                        <div className="flex flex-wrap gap-2">
                          {msg.sources.map((src, idx) => (
                            <div key={idx} className="bg-gray-800 border border-gray-700 rounded px-2 py-1 flex items-center gap-2 text-xs text-gray-300 cursor-help" title={src.preview}>
                              <FileText size={12} className="text-blue-400" suppressHydrationWarning />
                              <span>{src.file}</span>
                              {src.page && <span className="text-gray-500">pg {src.page}</span>}
                              <span className="text-emerald-500 bg-emerald-500/10 px-1 rounded">
                                {(src.score * 100).toFixed(0)}%
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="p-4 bg-gray-900 border-t border-gray-800 fixed bottom-0 left-0 right-0 md:absolute md:border-t-0 md:bg-transparent">
          <div className="max-w-3xl mx-auto bg-gray-800 rounded-xl border border-gray-700 shadow-xl overflow-hidden focus-within:border-gray-500 focus-within:ring-1 focus-within:ring-gray-500 transition-all">
            <form onSubmit={sendMessage} className="flex relative">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                  }
                }}
                placeholder={mode === "general" ? "Message Assistant..." : "Ask about your documents..."}
                className="w-full bg-transparent border-none p-4 pr-12 focus:outline-none resize-none min-h-[56px] max-h-32 text-gray-100 placeholder-gray-500"
                rows={1}
              />
              <button 
                type="submit" 
                disabled={!input.trim() || isStreaming}
                className="absolute right-2 bottom-2 p-2 rounded-lg bg-gray-700 text-white hover:bg-gray-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <Send size={18} suppressHydrationWarning />
              </button>
            </form>
          </div>
          <div className="text-center mt-2 text-xs text-gray-500">
            AI can make mistakes. Check important info.
          </div>
        </div>
      </div>
    </div>
  );
}
