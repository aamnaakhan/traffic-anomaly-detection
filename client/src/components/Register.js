import React, { useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import './Register.css';
import { AuthContext } from '../context/AuthContext';

function Register() {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [agreeTerms, setAgreeTerms] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const navigate = useNavigate();
  const { setUser } = useContext(AuthContext);

  const handleSubmit = (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    if (!agreeTerms) {
      setError('You must agree to the terms and conditions');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    if (!fullName || !email || !password) {
      setError('Please fill in all required fields');
      return;
    }

    // Simulate successful registration
    setSuccess('Account created successfully! Redirecting to login...');
    setUser({ email });
    setTimeout(() => {
      navigate('/login');
    }, 2000);
  };

  return (
    <div className="register-container">
      <h2>Create Account</h2>
      {error && <p className="error-message">{error}</p>}
      {success && <p className="success-message">{success}</p>}
      <form onSubmit={handleSubmit}>
        <label htmlFor="fullName">Full Name</label>
        <input
          id="fullName"
          type="text"
          value={fullName}
          onChange={e => setFullName(e.target.value)}
          required
          placeholder="Enter your full name"
        />
        <label htmlFor="email">Email</label>
        <input
          id="email"
          type="email"
          value={email}
          onChange={e => setEmail(e.target.value)}
          required
          placeholder="Enter your email"
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
        <label htmlFor="confirmPassword">Confirm Password</label>
        <input
          id="confirmPassword"
          type="password"
          value={confirmPassword}
          onChange={e => setConfirmPassword(e.target.value)}
          required
          placeholder="Confirm your password"
        />
        <div className="terms">
          <input
            id="agreeTerms"
            type="checkbox"
            checked={agreeTerms}
            onChange={e => setAgreeTerms(e.target.checked)}
          />
          <label htmlFor="agreeTerms">I agree to the terms and conditions</label>
        </div>
        <button type="submit" className="create-account-button">Create Account</button>
      </form>
      <div className="login-link" onClick={() => navigate('/login')}>
        Already have an account? Login
      </div>
    </div>
  );
}

export default Register;
