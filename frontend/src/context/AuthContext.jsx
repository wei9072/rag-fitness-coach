import { createContext, useContext, useState, useEffect } from 'react';
import { getMeAPI } from '../api/client';

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);       // { user_id, username }
  const [token, setToken] = useState(null);
  const [loading, setLoading] = useState(true);

  // 啟動時嘗試從 localStorage 恢復登入狀態
  useEffect(() => {
    const savedToken = localStorage.getItem('fitai_token');
    if (savedToken) {
      setToken(savedToken);
      // 驗證 token 是否還有效
      getMeAPI().then(res => {
        if (res.ok) {
          res.json().then(data => setUser(data));
        } else {
          // Token 過期，清除
          localStorage.removeItem('fitai_token');
          setToken(null);
        }
      }).catch(() => {
        localStorage.removeItem('fitai_token');
        setToken(null);
      }).finally(() => setLoading(false));
    } else {
      setLoading(false);
    }
  }, []);

  const login = (tokenStr, userData) => {
    localStorage.setItem('fitai_token', tokenStr);
    setToken(tokenStr);
    setUser(userData);
  };

  const logout = () => {
    localStorage.removeItem('fitai_token');
    setToken(null);
    setUser(null);
  };

  return (
    <AuthContext.Provider value={{
      user,
      token,
      isLoggedIn: !!user,
      loading,
      login,
      logout,
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
