import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { loginAPI, registerAPI } from '../api/client';

export default function LoginPage() {
  const [isRegister, setIsRegister] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      const res = isRegister
        ? await registerAPI(username, password)
        : await loginAPI(username, password);

      const data = await res.json();

      if (!res.ok) {
        setError(data.detail || '操作失敗');
        return;
      }

      login(data.access_token, {
        user_id: data.user_id,
        username: data.username,
      });
      navigate('/chat');
    } catch (err) {
      setError('無法連線到伺服器，請確認後端 API 是否啟動');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      {/* 背景光暈 */}
      <div className="glow glow-blue" />
      <div className="glow glow-green" />

      <div className="login-card">
        <div className="login-brand">
          <span className="brand-icon">💪</span>
          <h1>FitAI</h1>
          <p>智慧健身教練</p>
        </div>

        <div className="tab-switch">
          <button
            className={!isRegister ? 'active' : ''}
            onClick={() => { setIsRegister(false); setError(''); }}
          >
            登入
          </button>
          <button
            className={isRegister ? 'active' : ''}
            onClick={() => { setIsRegister(true); setError(''); }}
          >
            註冊
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <label htmlFor="username">帳號</label>
            <input
              id="username"
              type="text"
              placeholder="輸入你的帳號"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              minLength={2}
            />
          </div>

          <div className="input-group">
            <label htmlFor="password">密碼</label>
            <input
              id="password"
              type="password"
              placeholder="輸入你的密碼"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              minLength={4}
            />
          </div>

          {error && <div className="error-msg">{error}</div>}

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? '處理中...' : (isRegister ? '建立帳號' : '立即登入')}
          </button>
        </form>

        <button className="guest-btn" onClick={() => navigate('/chat')}>
          🚀 不登入，直接使用
        </button>
      </div>
    </div>
  );
}
