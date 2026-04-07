const API_BASE = import.meta.env.VITE_API_BASE || '';

/**
 * 通用 API 請求封裝
 * 自動帶入 JWT Token（如果存在）
 */
export async function apiRequest(endpoint, options = {}) {
  const token = localStorage.getItem('fitai_token');
  
  const headers = {
    'Content-Type': 'application/json',
    ...options.headers,
  };
  
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });
  
  return response;
}

/**
 * Auth API
 */
export async function registerAPI(username, password) {
  const res = await apiRequest('/api/auth/register', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
  return res;
}

export async function loginAPI(username, password) {
  const res = await apiRequest('/api/auth/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  });
  return res;
}

export async function getMeAPI() {
  const res = await apiRequest('/api/auth/me');
  return res;
}

/**
 * Chat API
 */
export async function chatAPI(question, sessionId = null, options = {}) {
  const body = {
    question,
    session_id: sessionId,
    api_key: options.api_key || null,
    llm_provider: options.llm_provider || 'groq',
    model_name: options.model_name || null,
    is_paid: options.is_paid || false,
    user_profile: options.user_profile || null,
  };
  
  const res = await apiRequest('/api/chat', {
    method: 'POST',
    body: JSON.stringify(body),
  });
  return res;
}
