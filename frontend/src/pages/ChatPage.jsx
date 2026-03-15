import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { chatAPI } from '../api/client';
import ReactMarkdown from 'react-markdown';

export default function ChatPage() {
  const { user, isLoggedIn, logout } = useAuth();
  const navigate = useNavigate();
  
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const now = () => new Date().toLocaleTimeString('zh-TW', { hour: '2-digit', minute: '2-digit' });

  const handleSend = async () => {
    const text = input.trim();
    if (!text || isLoading) return;

    const userMsg = { id: Date.now(), sender: 'user', text, time: now() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsLoading(true);

    try {
      const res = await chatAPI(text, sessionId);
      
      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `HTTP ${res.status}`);
      }

      const data = await res.json();
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        sender: 'ai',
        text: data.answer,
        sources: data.sources || [],
        rewrittenQuery: data.rewritten_query || '',
        time: now(),
      }]);
      
      if (data.session_id) {
        setSessionId(data.session_id);
      }
    } catch (err) {
      const isNetwork = err.message === 'Failed to fetch';
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        sender: 'ai',
        text: isNetwork
          ? '⚠️ 連線失敗，後端 API 尚未啟動。\n請確認 backend `python main.py` 正在運行中。'
          : `⚠️ ${err.message}`,
        time: now(),
      }]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleNewChat = () => {
    setMessages([]);
    setSessionId(null);
  };

  return (
    <div className="chat-page">
      {/* 背景光暈 */}
      <div className="glow glow-blue" />
      <div className="glow glow-green" />

      {/* Header */}
      <header className="chat-header">
        <div className="header-left">
          <button className="icon-btn" onClick={() => setSidebarOpen(!sidebarOpen)}>☰</button>
          <h1>💪 FitAI</h1>
        </div>
        <div className="header-right">
          {isLoggedIn ? (
            <>
              <span className="user-badge">👤 {user.username}</span>
              <button className="text-btn" onClick={logout}>登出</button>
            </>
          ) : (
            <button className="text-btn" onClick={() => navigate('/')}>登入</button>
          )}
        </div>
      </header>

      {/* Sidebar */}
      {sidebarOpen && (
        <div className="sidebar-overlay" onClick={() => setSidebarOpen(false)}>
          <div className="sidebar" onClick={(e) => e.stopPropagation()}>
            <h3>⚙️ 設定</h3>
            <button className="sidebar-btn" onClick={handleNewChat}>🔄 開啟新對話</button>
            {!isLoggedIn && (
              <div className="sidebar-notice">
                <p>💡 目前為匿名模式</p>
                <p>登入後 AI 教練會記住你的訓練紀錄喔！</p>
                <button className="submit-btn" onClick={() => navigate('/')}>前往登入</button>
              </div>
            )}
            {isLoggedIn && (
              <div className="sidebar-notice success">
                <p>✅ 已登入為 <strong>{user.username}</strong></p>
                <p>你的對話紀錄將被儲存，下次回來可以接續聊天！</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Messages */}
      <main className="chat-messages">
        {messages.length === 0 && (
          <div className="welcome">
            <div className="welcome-icon">🏋️</div>
            <h2>歡迎使用 FitAI 智慧健身教練</h2>
            <p>你可以詢問訓練紀錄、安排訓練菜單，或是請教任何健身相關的問題！</p>
            {!isLoggedIn && (
              <p className="welcome-hint">💡 <button className="inline-link" onClick={() => navigate('/')}>登入</button> 後，AI 教練可以根據你的歷史紀錄做出個人化回答</p>
            )}
            <div className="quick-actions">
              {['我最近深蹲做多重？', '幫我安排一週訓練菜單', '如何避免訓練受傷？'].map(q => (
                <button key={q} className="quick-btn" onClick={() => { setInput(q); }}>
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map(msg => (
          <div key={msg.id} className={`msg-row ${msg.sender}`}>
            <div className={`msg-bubble ${msg.sender}`}>
              {msg.sender === 'ai' ? (
                <ReactMarkdown>{msg.text}</ReactMarkdown>
              ) : (
                <p>{msg.text}</p>
              )}
              <span className="msg-time">{msg.time}</span>
              
              {msg.sources && msg.sources.length > 0 && (
                <details className="sources-detail">
                  <summary>📚 參考 {msg.sources.length} 筆紀錄</summary>
                  <div className="sources-list">
                    {msg.sources.map((s, i) => (
                      <div key={i} className="source-chip">{s.substring(0, 120)}...</div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          </div>
        ))}

        {isLoading && (
          <div className="msg-row ai">
            <div className="msg-bubble ai typing">
              <span className="dot" /><span className="dot" /><span className="dot" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </main>

      {/* Input */}
      <footer className="chat-input-bar">
        <div className="input-wrapper">
          <textarea
            ref={inputRef}
            rows="1"
            placeholder="輸入你的問題..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
          <button className="send-btn" onClick={handleSend} disabled={isLoading || !input.trim()}>
            ➤
          </button>
        </div>
      </footer>
    </div>
  );
}
