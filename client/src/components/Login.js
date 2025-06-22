import React, { useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import './Login.css';
import { AuthContext } from '../context/AuthContext';

function Login() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();
  const { setUser } = useContext(AuthContext);

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');
    if (!email || !password) {
      setError('Please enter email and password');
      return;
    }
    // Simulate successful login
    localStorage.setItem('token', 'dummy-token');
    const userObj = { email };
    localStorage.setItem('user', JSON.stringify(userObj));
    setUser(userObj);
    navigate('/dashboard');
  };

  return (
    <div className="login-container">
      <h2>Sign In</h2>
      {error && <p className="error-message">{error}</p>}
      <form onSubmit={handleSubmit}>
        <label htmlFor="email">Email or Username</label>
        <input
          id="email"
          type="email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
          placeholder="Enter your email or username"
        />
        <label htmlFor="password">Password</label>
        <input
          id="password"
          type="password"
          value={password}
          onChange={e => setPassword(e.target.value)}
          required
          placeholder="Enter your password"
        />
        <div className="forgot-password" onClick={() => alert('Forgot Password clicked')}>
          Forgot Password?
        </div>
        <button type="submit" className="sign-in-button">Sign In</button>
      </form>
      <div className="create-account-link" onClick={() => navigate('/register')}>
        Don't have an account? Create Account
      </div>
    </div>
  );
}

export default Login;
