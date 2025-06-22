import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import './App.css';
import './components/HomeEnhanced.css';
import NavBar from './components/NavBar';
import RecordedFootageDetection from './components/RecordedFootageDetection';
import Login from './components/Login';
import Dashboard from './components/Dashboard';
import Register from './components/Register';
import { AuthProvider } from './context/AuthContext';
import HomeEnhanced from './components/HomeEnhanced';

function PrivateRoute({ children }) {
  const token = localStorage.getItem('token');
  return token ? children : <Navigate to="/login" />;
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <NavBar />
        <Routes>
          <Route path="/" element={<HomeEnhanced />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />
          <Route path="/dashboard" element={
            <PrivateRoute>
              <Dashboard />
            </PrivateRoute>
          } />
          <Route path="/recorded-detect" element={<RecordedFootageDetection />} />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
