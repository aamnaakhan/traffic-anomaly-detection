// ✨ Updated Full Version of RecordedFootageDetection.js with Vehicle Count Support

import React, { useState, useEffect, useRef } from 'react';
import ReactPlayer from 'react-player';
import './RecordedFootageDetection.css';

function RecordedFootageDetection() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileType, setFileType] = useState('');
  const [videoUrl, setVideoUrl] = useState('');
  const [imageUrl, setImageUrl] = useState('');
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [showPreview, setShowPreview] = useState(false);
  const [outputFile, setOutputFile] = useState('');
  const [timestamp, setTimestamp] = useState('');
  const [outputFilesList, setOutputFilesList] = useState([]);
  const [isDragging, setIsDragging] = useState(false);
  const [region, setRegion] = useState(null);
  const [region2, setRegion2] = useState(null);
  const [region3, setRegion3] = useState(null);
  const [startPoint, setStartPoint] = useState(null);
  const [startPoint2, setStartPoint2] = useState(null);
  const [startPoint3, setStartPoint3] = useState(null);
  const [saveMessage, setSaveMessage] = useState('');
  const [detectionType, setDetectionType] = useState(['wrong_way']);
  const [linePoints, setLinePoints] = useState([]);
  const [lineStep, setLineStep] = useState(0);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const handleSaveFootpath = async () => {
    if (!region3) {
      setSaveMessage('Please select the footpath region first.');
      return;
    }
    // Calculate scale factor between displayed image and original frame size using image natural size
    let scaleX = 1;
    let scaleY = 1;
    const img = document.querySelector('img[alt="Selected Frame"]');
    if (img && img.naturalWidth && img.naturalHeight) {
      const displayedWidth = img.clientWidth;
      const displayedHeight = img.clientHeight;
      scaleX = img.naturalWidth / displayedWidth;
      scaleY = img.naturalHeight / displayedHeight;
      console.log('Scale factors:', scaleX, scaleY);
    }
    // Scale coordinates to original frame size
    const x1 = Math.round(region3.x * scaleX);
    const y1 = Math.round(region3.y * scaleY);
    const x2 = Math.round((region3.x + region3.width) * scaleX);
    const y2 = Math.round((region3.y + region3.height) * scaleY);
    const coords = [
      [x1, y1],
      [x2, y1],
      [x2, y2],
      [x1, y2],
    ];
    console.log('Scaled coords:', coords);
    try {
      const response = await fetch('http://localhost:8000/set_footpath', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ footpath_polygons: [coords] }),
      });
      const data = await response.json();
      if (response.ok) {
        setSaveMessage('Footpath saved successfully.');
      } else {
        setSaveMessage('Error saving footpath: ' + (data.error || 'Unknown error'));
      }
    } catch (error) {
      setSaveMessage('Error saving footpath: ' + error.message);
    }
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;
    setSelectedFile(file);
    setVideoUrl('');
    setImageUrl('');
    setSelectedFrame(null);
    setAnomalies([]);
    setError('');
    setShowPreview(false);
    setOutputFile('');
    setTimestamp('');
    setRegion(null);
    setRegion2(null);
    setStartPoint(null);
    setStartPoint2(null);
    setRegion3(null);
    setStartPoint3(null);
    setLinePoints([]);
    setLineStep(0);
    setSaveMessage('');
    setDetectionType(['wrong_way']);
    let type = '';
    if (file.type && file.type !== '') {
      type = file.type.startsWith('video') ? 'video' : file.type.startsWith('image') ? 'image' : '';
    } else {
      const ext = file.name.split('.').pop().toLowerCase();
      if (["mp4", "m4v", "avi", "mov", "mkv", "flv", "wmv"].includes(ext)) {
        type = 'video';
      } else if (["jpg", "jpeg", "png", "gif", "bmp"].includes(ext)) {
        type = 'image';
      }
    }
    setFileType(type);
    const url = URL.createObjectURL(file);
    if (type === 'video') setVideoUrl(url);
    else if (type === 'image') {
      setImageUrl(url);
      setSelectedFrame(url);
    }
  };

  const captureFrame = () => {
    if (!videoRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/png');
    setSelectedFrame(dataUrl);
  };

  const handleMouseDown = (e) => {
    if (!selectedFrame) return;
    const { left, top } = e.target.getBoundingClientRect();
    const x = e.clientX - left;
    const y = e.clientY - top;
  
    if (!region && detectionType.includes('wrong_way')) {
      setStartPoint({ x, y });
      setRegion({ x, y, width: 0, height: 0 });
      setIsDragging(true);
      return;
    }
  
    if (!region2 && detectionType.includes('prohibited_area')) {
      setStartPoint2({ x, y });
      setRegion2({ x, y, width: 0, height: 0 });
      setIsDragging(true);
      return;
    }
  
    if (!region3 && detectionType.includes('footpath')) {
      setStartPoint3({ x, y });
      setRegion3({ x, y, width: 0, height: 0 });
      setIsDragging(true);
      return;
    }
  
    if (detectionType.includes('vehicle_count') && linePoints.length < 4) {
      setLinePoints((prev) => [...prev, { x, y }]);
      return;
    }
  };
  

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    const { left, top } = e.target.getBoundingClientRect();
    const x = e.clientX - left;
    const y = e.clientY - top;
  
    if (startPoint && !region2 && !region3) {
      setRegion({
        x: Math.min(x, startPoint.x),
        y: Math.min(y, startPoint.y),
        width: Math.abs(x - startPoint.x),
        height: Math.abs(y - startPoint.y)
      });
    } else if (startPoint2 && !region3) {
      setRegion2({
        x: Math.min(x, startPoint2.x),
        y: Math.min(y, startPoint2.y),
        width: Math.abs(x - startPoint2.x),
        height: Math.abs(y - startPoint2.y)
      });
    } else if (startPoint3) {
      setRegion3({
        x: Math.min(x, startPoint3.x),
        y: Math.min(y, startPoint3.y),
        width: Math.abs(x - startPoint3.x),
        height: Math.abs(y - startPoint3.y)
      });
    }
  };
  


  const handleMouseUp = () => {
    setIsDragging(false);
  };
  

  const handleSaveVehicleLines = async () => {
    console.log("linePoints before saving:", linePoints);
    if (linePoints.length !== 4) {
      setSaveMessage('Please draw both lines (2 points each).');
      return;
    }
  
    const img = document.querySelector('img[alt="Selected Frame"]');
    if (!img || !img.naturalWidth || !img.naturalHeight) {
      setSaveMessage('Could not get video resolution for scaling.');
      return;
    }
  
    const width = img.naturalWidth;
    const height = img.naturalHeight;
  
    const scaleX = img.naturalWidth / img.clientWidth;
    const scaleY = img.naturalHeight / img.clientHeight;

    const scalePoint = (pt) => [Math.round(pt.x * scaleX), Math.round(pt.y * scaleY)];

    const rawLine1 = [scalePoint(linePoints[0]), scalePoint(linePoints[1])];
    const rawLine2 = [scalePoint(linePoints[2]), scalePoint(linePoints[3])];

  
    console.log("Sending raw lines:", rawLine1, rawLine2, width, height);
  
    try {
      const res = await fetch('http://localhost:8000/set_lines', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          line1: rawLine1,
          line2: rawLine2,
          video_width: width,
          video_height: height,
        }),
      });
      const data = await res.json();
      if (res.ok) setSaveMessage('Vehicle lines saved successfully.');
      else setSaveMessage('Error saving lines: ' + data.error);
    } catch (err) {
      setSaveMessage('Error saving lines: ' + err.message);
    }
  };
  
  


  const handleSaveGateZone = async () => {
    if (!region) {
      setSaveMessage('Please select a region first.');
      return;
    }
    // Calculate scale factor between displayed image and original frame size using image natural size
    let scaleX = 1;
    let scaleY = 1;
    const img = document.querySelector('img[alt="Selected Frame"]');
    if (img && img.naturalWidth && img.naturalHeight) {
      const displayedWidth = img.clientWidth;
      const displayedHeight = img.clientHeight;
      scaleX = img.naturalWidth / displayedWidth;
      scaleY = img.naturalHeight / displayedHeight;
      console.log('Scale factors:', scaleX, scaleY);
    }
    // Scale coordinates to original frame size
    const x1 = Math.round(region.x * scaleX);
    const y1 = Math.round(region.y * scaleY);
    const x2 = Math.round((region.x + region.width) * scaleX);
    const y2 = Math.round((region.y + region.height) * scaleY);
    const coords = [x1, y1, x2, y2];
    console.log('Scaled coords:', coords);
    try {
      const response = await fetch('http://localhost:8000/set_gate_zone', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gate_zone: coords }),
      });
      const data = await response.json();
      if (response.ok) {
        setSaveMessage('Gate zone saved successfully.');
      } else {
        setSaveMessage('Error saving gate zone: ' + (data.error || 'Unknown error'));
      }
    } catch (error) {
      setSaveMessage('Error saving gate zone: ' + error.message);
    }
  };

  const handleSaveProhibitedArea = async () => {
    if (!region2) {
      setSaveMessage('Please select the prohibited area region first.');
      return;
    }
    // Calculate scale factor between displayed image and original frame size using image natural size
    let scaleX = 1;
    let scaleY = 1;
    const img = document.querySelector('img[alt="Selected Frame"]');
    if (img && img.naturalWidth && img.naturalHeight) {
      const displayedWidth = img.clientWidth;
      const displayedHeight = img.clientHeight;
      scaleX = img.naturalWidth / displayedWidth;
      scaleY = img.naturalHeight / displayedHeight;
      console.log('Scale factors:', scaleX, scaleY);
    }
    // Scale coordinates to original frame size
    const x1 = Math.round(region2.x * scaleX);
    const y1 = Math.round(region2.y * scaleY);
    const x2 = Math.round((region2.x + region2.width) * scaleX);
    const y2 = Math.round((region2.y + region2.height) * scaleY);
    const coords = [x1, y1, x2, y2];
    console.log('Scaled coords:', coords);
    try {
      const response = await fetch('http://localhost:8000/set_prohibited_area', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prohibited_area: coords }),
      });
      const data = await response.json();
      if (response.ok) {
        setSaveMessage('Prohibited area saved successfully.');
      } else {
        setSaveMessage('Error saving prohibited area: ' + (data.error || 'Unknown error'));
      }
    } catch (error) {
      setSaveMessage('Error saving prohibited area: ' + error.message);
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError('Please select a video file first.');
      return;
    }
    setLoading(true);
    setError('');
    setShowPreview(false);
    const formData = new FormData();
    formData.append('video', selectedFile);
    formData.append('detection_type', detectionType);

    try {
      const response = await fetch('http://localhost:8000/detect', {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        throw new Error('Failed to process video');
      }
      const data = await response.json();
      setAnomalies(data.anomalies || []);
      setVideoUrl(data.video_url);
      setOutputFile(data.output_file || '');
      setTimestamp(data.timestamp || '');
      setShowPreview(true);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const fetchOutputFiles = async () => {
    try {
      const response = await fetch('http://localhost:8000/list_output_files');
      if (!response.ok) {
        throw new Error('Failed to fetch output files');
      }
      const data = await response.json();
      setOutputFilesList(data.files || []);
    } catch (err) {
      setError(err.message);
    }
  };

  useEffect(() => {
    fetchOutputFiles();
  }, []);

  return (
    <div className="container">
      <h2>Recorded Footage Detection and Annotation</h2>
      <input type="file" accept="video/*,image/*,.mp4,.m4v" onChange={handleFileChange} />
      {fileType === 'video' && (
        <div style={{ marginTop: '10px' }}>
          <video
            ref={videoRef}
            src={videoUrl}
            controls
            width="600"
            onPause={captureFrame}
            onSeeked={captureFrame}
          />
          <p>Use the video controls to select the starting frame. Pause or seek to capture the frame.</p>
        </div>
      )}

      {fileType === 'video' && (
        <div style={{ marginTop: '10px' }}>
          <div className="detection-types">
            <label>
              <input
                type="checkbox"
                value="wrong_way"
                checked={detectionType.includes('wrong_way')}
                onChange={(e) => {
                  const checked = e.target.checked;
                  setDetectionType((prev) => {
                    if (checked) {
                      return [...prev, 'wrong_way'];
                    } else {
                      return prev.filter((item) => item !== 'wrong_way');
                    }
                  });
                }}
              />
              Wrong Way Detection
            </label>
            <label>
              <input
                type="checkbox"
                value="prohibited_area"
                checked={detectionType.includes('prohibited_area')}
                onChange={(e) => {
                  const checked = e.target.checked;
                  setDetectionType((prev) => {
                    if (checked) {
                      return [...prev, 'prohibited_area'];
                    } else {
                      return prev.filter((item) => item !== 'prohibited_area');
                    }
                  });
                }}
              />
              Prohibited Area Detection
            </label>
            <label>
              <input
                type="checkbox"
                value="footpath"
                checked={detectionType.includes('footpath')}
                onChange={(e) => {
                  const checked = e.target.checked;
                  setDetectionType((prev) => {
                    if (checked) {
                      return [...prev, 'footpath'];
                    } else {
                      return prev.filter((item) => item !== 'footpath');
                    }
                  });
                }}
              />
              Footpath Detection
            </label>
            <label>
              <input
                type="checkbox"
                value="vehicle_count"
                checked={detectionType.includes('vehicle_count')}
                onChange={(e) => {
                  const checked = e.target.checked;
                  setDetectionType((prev) =>
                    checked ? [...prev, 'vehicle_count'] : prev.filter((t) => t !== 'vehicle_count')
                  );
                }}
              />
              Vehicle Count Detection
            </label>

            
            

      {selectedFrame && (
        <div style={{ position: 'relative', marginTop: '20px', display: 'inline-block' }}>
          <img
            src={selectedFrame}
            alt="Selected Frame"
            style={{ maxWidth: '600px', userSelect: 'none' }}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          />
          {linePoints.map((pt, idx) => (
            <div
                key={idx}
                style={{
                position: 'absolute',
                left: pt.x - 4,
                top: pt.y - 4,
                width: 8,
                height: 8,
                backgroundColor: 'purple',
                borderRadius: '50%',
                pointerEvents: 'none',
                zIndex: 20,
                }}
            />
            ))}

            {linePoints.length === 4 && (
            <svg
                style={{
                position: 'absolute',
                top: 0,
                left: 0,
                width: '600px',
                height: 'auto',
                zIndex: 15,
                }}
            >
                <line
                x1={linePoints[0].x}
                y1={linePoints[0].y}
                x2={linePoints[1].x}
                y2={linePoints[1].y}
                stroke="blue"
                strokeWidth="2"
                />
                <line
                x1={linePoints[2].x}
                y1={linePoints[2].y}
                x2={linePoints[3].x}
                y2={linePoints[3].y}
                stroke="green"
                strokeWidth="2"
                />
            </svg>
            )}


          {region && (
            <div
              style={{
                position: 'absolute',
                border: '2px dashed red',
                left: region.x,
                top: region.y,
                width: region.width,
                height: region.height,
                pointerEvents: 'none',
                zIndex: 10,
              }}
            >
              <span
                style={{
                  position: 'absolute',
                  top: -20,
                  left: 0,
                  color: 'red',
                  fontWeight: 'bold',
                  backgroundColor: 'white',
                  padding: '0 4px',
                  fontSize: '12px',
                  userSelect: 'none',
                }}
              >
                Gate Zone
              </span>
            </div>
          )}
          {detectionType.includes('prohibited_area') && region2 && (
            <div
              style={{
                position: 'absolute',
                border: '2px dashed blue',
                left: region2.x,
                top: region2.y,
                width: region2.width,
                height: region2.height,
                pointerEvents: 'none',
                zIndex: 10,
              }}
            >
              <span
                style={{
                  position: 'absolute',
                  top: -20,
                  left: 0,
                  color: 'blue',
                  fontWeight: 'bold',
                  backgroundColor: 'white',
                  padding: '0 4px',
                  fontSize: '12px',
                  userSelect: 'none',
                }}
              >
                Prohibited Area
              </span>
            </div>
          )}
          {detectionType.includes('footpath') && region3 && (
            <div
              style={{
                position: 'absolute',
                border: '2px dashed green',
                left: region3.x,
                top: region3.y,
                width: region3.width,
                height: region3.height,
                pointerEvents: 'none',
                zIndex: 10,
              }}
            >
              <span
                style={{
                  position: 'absolute',
                  top: -20,
                  left: 0,
                  color: 'green',
                  fontWeight: 'bold',
                  backgroundColor: 'white',
                  padding: '0 4px',
                  fontSize: '12px',
                  userSelect: 'none',
                }}
              >
                Footpath
              </span>
            </div>
          )}
        </div>
      )}
      <canvas ref={canvasRef} style={{ display: 'none' }} />
      {detectionType.includes('wrong_way') && (
        <button onClick={handleSaveGateZone} style={{ marginTop: '20px' }}>
          Save Gate Zone
        </button>
      )}
      {detectionType.includes('prohibited_area') && (
        <>
          <button onClick={handleSaveProhibitedArea} style={{ marginTop: '20px' }}>
            Save Prohibited Area
          </button>
          {region2 && (
            <p style={{ marginTop: '10px', color: 'green' }}>Prohibited area region selected</p>
          )}
        </>
      )}
      {detectionType.includes('vehicle_count') && (
                <>
                    <button onClick={handleSaveVehicleLines} style={{ marginTop: '10px' }}>
                    Save Vehicle Count Lines
                    </button>
                    <p style={{ color: 'purple' }}>
                    Click 4 points to draw 2 lines (UP and DOWN)
                    </p>
                </>
                )}
          </div>
        </div>
      )}
      {detectionType.includes('footpath') && (
        <>
          <button onClick={handleSaveFootpath} style={{ marginTop: '20px' }}>
            Save Footpath
          </button>
          {region3 && (
            <p style={{ marginTop: '10px', color: 'green' }}>Footpath region selected</p>
          )}
        </>
      )}
      <button onClick={handleUpload} disabled={loading} style={{ marginLeft: '10px' }}>
        {loading ? 'Processing...' : 'Upload and Detect'}
      </button>
      {saveMessage && <p style={{ marginTop: '10px' }}>{saveMessage}</p>}
      {error && <p className="error-message">{error}</p>}

      <h3>Output Files</h3>
      <button className="output-files" onClick={fetchOutputFiles}>Refresh Output Files</button>
      <div className="output-files">
        <select
          onChange={(e) => setVideoUrl(`http://localhost:8000/processed_video/${e.target.value}`)}
          defaultValue=""
          style={{ padding: '8px 12px', borderRadius: '6px', fontSize: '15px', width: '100%' }}
        >
          <option value="" disabled>Select an output file</option>
          {outputFilesList.map((file) => (
            <option key={file} value={file}>
              {file}
            </option>
          ))}
        </select>
      </div>

      {anomalies.length > 0 && (
        <div>
          <h3>Detected Anomalies</h3>
          <ul className="anomalies-list">
            {anomalies.map((anomaly) => (
              <li key={anomaly.id}>
                ID: {anomaly.id}, Class: {anomaly.class}, Description: {anomaly.description}
              </li>
            ))}
          </ul>
          <p>
            Output File: <a href={`http://localhost:8000/processed_video/${outputFile}`} target="_blank" rel="noopener noreferrer">{outputFile}</a><br />
            Saved at: {timestamp}
          </p>
        </div>
      )}
      {videoUrl && (
        <div className="video-preview">
          <h3>Video Preview</h3>
          <p className="video-url">Video URL: {videoUrl}</p>
          <ReactPlayer url={videoUrl} controls width="100%" />
        </div>
      )}
    </div>
  );
}

export default RecordedFootageDetection;
