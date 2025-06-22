import React, { useState, useEffect } from 'react';
import { NavLink, useNavigate } from 'react-router-dom';
import './NavBar.css';

function NavBar() {
  const [user, setUser] = useState(null);
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    const storedUser = localStorage.getItem('user');
    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }
  }, []);

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
    setUser(null);
    setDropdownOpen(false);
    navigate('/');
  };

  const toggleDropdown = () => {
    setDropdownOpen(!dropdownOpen);
  };

  return (
    <nav className="navbar">
      <div className="navbar-logo">Traffic Anomaly Detection System</div>
      <ul className="navbar-links">
        <li><NavLink to="/" activeClassName="active-link" exact={true}>Home</NavLink></li>
        {user ? (
          <>
            <li><NavLink to="/dashboard" activeClassName="active-link">Dashboard</NavLink></li>
            <li><NavLink to="/recorded-detect" activeClassName="active-link">Recorded Footage Detection</NavLink></li>
            <li className="navbar-user" onClick={toggleDropdown}>
              Hello, {user.email} <span className="dropdown-arrow">{dropdownOpen ? '▲' : '▼'}</span>
              {dropdownOpen && (
                <ul className="dropdown-menu">
                  <li onClick={() => { setDropdownOpen(false); navigate('/profile'); }}>Profile</li>
                  <li onClick={handleLogout}>Sign Out</li>
                </ul>
              )}
            </li>
          </>
        ) : (
          <>
            <li><NavLink to="/login" activeClassName="active-link">Login</NavLink></li>
            <li><NavLink to="/register" activeClassName="active-link">Create Account</NavLink></li>
            <li><NavLink to="/recorded-detect" activeClassName="active-link">Recorded Footage Detection</NavLink></li>
          </>
        )}
      </ul>
    </nav>
  );
}

export default NavBar;
