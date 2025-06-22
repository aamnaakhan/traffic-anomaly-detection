import React, { useState, useEffect } from 'react';
import './Dashboard.css';

function Dashboard() {
  const [darkMode, setDarkMode] = useState(true);
  const [activeMenu, setActiveMenu] = useState('dashboard');
  const alerts = [
    { id: 1, type: 'Collision', description: 'Collision detected at 5th Avenue', time: '2 mins ago' },
    { id: 2, type: 'Wrong-way Driving', description: 'Vehicle driving wrong way on Main St.', time: '10 mins ago' },
    { id: 3, type: 'Over-speeding', description: 'Speeding detected on Highway 101', time: '30 mins ago' },
  ];
  const [themeClass, setThemeClass] = useState('dark-mode');

  useEffect(() => {
    document.body.className = themeClass;
  }, [themeClass]);

  const toggleDarkMode = () => {
    if (darkMode) {
      setThemeClass('light-mode');
    } else {
      setThemeClass('dark-mode');
    }
    setDarkMode(!darkMode);
  };

  const menuItems = [
    { key: 'dashboard', label: 'Dashboard' },
    // Removed Live Feed from sidebar menu as requested
    { key: 'alerts', label: 'Alerts' },
    { key: 'reports', label: 'Reports' },
    { key: 'settings', label: 'Settings' },
  ];

  return (
    <div className="dashboard-container">
      <aside className="sidebar">
        <div className="logo">Traffic ADS</div>
        <nav>
          <ul>
            {menuItems.map(item => (
              <li
                key={item.key}
                className={activeMenu === item.key ? 'active' : ''}
                onClick={() => setActiveMenu(item.key)}
              >
                <span>{item.label}</span>
              </li>
            ))}
          </ul>
        </nav>
        <button className="toggle-theme" onClick={toggleDarkMode}>
          {darkMode ? 'Light Mode' : 'Dark Mode'}
        </button>
      </aside>
      <div className="main-content">
        <header className="topbar">
          <div className="status">System Status: Online</div>
          <div className="search-bar">
            <input type="text" placeholder="Search..." />
          </div>
          <div className="user-profile">
            <img src="https://i.pravatar.cc/36" alt="User Profile" />
          </div>
        </header>

        {activeMenu === 'dashboard' && (
          <>
            {/* Live Traffic Camera Feed section removed as per user request */}

            <section className="section">
              <h2>Recent Traffic Anomalies</h2>
              <div className="card alert-panel">
                {alerts.map(alert => (
                  <div key={alert.id} className="alert-item">
                    <strong>{alert.type}:</strong> {alert.description} <em>({alert.time})</em>
                  </div>
                ))}
              </div>
            </section>

            <section className="section charts">
              <div className="card chart-card">
                <h3>Historical Anomaly Trends (Bar Chart)</h3>
                {/* Placeholder for bar chart */}
                <div style={{height: '180px', color: '#ccc', display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
                  Bar chart goes here.
                </div>
              </div>
              <div className="card chart-card">
                <h3>Violation Types (Pie Chart)</h3>
                {/* Placeholder for pie chart */}
                <div style={{height: '180px', color: '#ccc', display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
                  Pie chart goes here.
                </div>
              </div>
              <div className="card chart-card">
                <h3>Traffic Flow (Line Chart)</h3>
                {/* Placeholder for line chart */}
                <div style={{height: '180px', color: '#ccc', display: 'flex', justifyContent: 'center', alignItems: 'center'}}>
                  Line chart goes here.
                </div>
              </div>
            </section>

            <section className="section map-section">
              <h2>Detected Anomaly Locations</h2>
              {/* Placeholder for map */}
              Map with red markers will appear here.
            </section>
          </>
        )}

        {activeMenu === 'livefeed' && (
          <section className="section">
            <h2>Live Feed</h2>
            <div className="card live-feed">
              Live feed content will be implemented here.
            </div>
          </section>
        )}

        {activeMenu === 'alerts' && (
          <section className="section">
            <h2>Alerts</h2>
            <div className="card alert-panel">
              {alerts.map(alert => (
                <div key={alert.id} className="alert-item">
                  <strong>{alert.type}:</strong> {alert.description} <em>({alert.time})</em>
                </div>
              ))}
            </div>
          </section>
        )}

        {activeMenu === 'reports' && (
          <section className="section">
            <h2>Reports</h2>
            <div className="card">
              Reports content will be implemented here.
            </div>
          </section>
        )}

        {activeMenu === 'settings' && (
          <section className="section">
            <h2>Settings</h2>
            <div className="card">
              Settings content will be implemented here.
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

export default Dashboard;
