import React, { useContext } from 'react';
import { AuthContext } from '../context/AuthContext';
import './HomeEnhanced.css';
import { FaVideo, FaCogs, FaExclamationTriangle, FaCheckCircle, FaProjectDiagram } from 'react-icons/fa';

const HomeEnhanced = () => {
  const { user } = useContext(AuthContext);

  return (
    <div className="home-enhanced-container">
      {/* Hero Section */}
      <section className="hero-section">
        <h1 className="hero-title">Traffic Anomaly Detection System</h1>
        <p className="hero-subtitle">Real-time monitoring and analysis for safer roads and efficient traffic management.</p>
        {user && (
          <button className="cta-button" onClick={() => window.location.href = '/dashboard'}>
            Get Started
          </button>
        )}
      </section>

      {/* Features Section */}
      <section className="features-section">
        <div className="feature-card" tabIndex="0">
          <FaVideo className="feature-icon" />
          <h3>Recorded Footage Analysis <span className="ai-tag">Upload & Analyze</span></h3>
          <p>Upload and analyze recorded video footage to identify traffic incidents and improve safety measures.</p>
          <button className="upload-button" onClick={() => alert('Upload Video clicked')}>
            Upload Video
          </button>
        </div>
        <div className="feature-card" tabIndex="0">
          <FaCogs className="feature-icon" />
          <h3>Customizable Detection Models <span className="ai-tag">AI-powered</span></h3>
          <p>Choose from multiple detection models or customize your own to suit specific traffic monitoring needs.</p>
        </div>
      </section>

      {/* Problem Domain Section */}
      <section className="info-section problem-domain">
        <h2><FaExclamationTriangle className="section-icon" /> Problem Domain</h2>
        <p>
          The primary challenge in traffic surveillance is to detect and respond to anomalous behavior in real time. These anomalies can include wrong-way driving, over-speeding, unauthorized entry into restricted zones, and pedestrians walking off designated footpaths. In addition to these violations, the system must be capable of identifying emergency vehicles such as ambulances, fire trucks, and police cars, and ensure that normal traffic rules are not incorrectly applied to them. Manual traffic monitoring systems are often inconsistent and prone to human error, making it difficult to detect such incidents with accuracy and speed. As urban areas become more congested, the reliance on human operators becomes increasingly impractical, leading to delayed responses and oversight. Therefore, there is a growing need for intelligent, automated systems that can provide continuous, real-time monitoring and reliable anomaly detection to enhance road safety and improve overall traffic management.
        </p>
      </section>

      {/* Problem Statement Section */}
      <section className="info-section problem-statement">
        <h2>Problem Statement</h2>
        <ul>
          <li><FaCheckCircle className="check-icon" /> Detect traffic violations (wrong-way driving, footpath misuse, etc.)</li>
          <li><FaCheckCircle className="check-icon" /> Track vehicles/pedestrians in real time</li>
          <li><FaCheckCircle className="check-icon" /> Detect accidents visually and with trajectories</li>
          <li><FaCheckCircle className="check-icon" /> Identify emergency vehicles</li>
          <li><FaCheckCircle className="check-icon" /> User interface for uploading and analyzing videos</li>
        </ul>
      </section>

      {/* Objectives Section */}
      <section className="info-section objectives">
        <h2>Project Objectives</h2>
        <ul className="two-column-list">
          <li><FaCheckCircle className="check-icon" /> Detect anomalies using computer vision</li>
          <li><FaCheckCircle className="check-icon" /> Use YOLOv10s for object detection</li>
          <li><FaCheckCircle className="check-icon" /> Track using Kalman filters</li>
          <li><FaCheckCircle className="check-icon" /> Exempt emergency vehicles from rule violations</li>
          <li><FaCheckCircle className="check-icon" /> Detect accidents via two approaches</li>
          <li><FaCheckCircle className="check-icon" /> Provide dashboard for result review</li>
        </ul>
      </section>

      {/* Scope Section */}
      <section className="info-section scope">
        <h2><FaProjectDiagram className="section-icon" /> Scope of the Project</h2>
        <ul className="two-column-list">
          <li>Detect vehicles, persons, helmets, emergency vehicles</li>
          <li>Identify traffic violations (wrong-way, prohibited zones)</li>
          <li>Accident detection via visual and trajectory-based systems</li>
          <li>Exempt emergency vehicles from applicable rules</li>
          <li>Upload interface with annotated results</li>
          <li>Support for video input, future real-time stream support</li>
        </ul>
      </section>
    </div>
  );
};

export default HomeEnhanced;
