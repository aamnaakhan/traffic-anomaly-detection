const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');
const User = require('./models/user');  // Import User model
const { authenticateToken, registerUser, loginUser, googleLogin } = require('./auth');
const app = express();
const port = 5000;

// MongoDB Atlas connection string (replace <username>, <password>, and myDatabase)
const mongoURI = 'mongodb+srv://syedz7169:jX8TCNRXUEf8rqWM@cluster0.znzqo3f.mongodb.net/traffic_anomaly_detection';

// Connect to MongoDB Atlas
mongoose.connect(mongoURI, {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
  .then(() => console.log('Connected to MongoDB Atlas'))
  .catch(err => console.error('Error connecting to MongoDB Atlas:', err));

// Middleware to parse JSON bodies
app.use(cors({ origin: 'http://localhost:3000' }));
app.use(express.json());

// Sample route (GET and POST for users)
app.get('/users', async (req, res) => {
  try {
    const users = await User.find();
    res.status(200).json(users);
  } catch (err) {
    res.status(400).json({ message: 'Error fetching users', error: err });
  }
});

app.post('/users', async (req, res) => {
  try {
    const { name, email, age } = req.body;
    const newUser = new User({ name, email, age });
    await newUser.save();
    res.status(201).json(newUser);
  } catch (err) {
    res.status(400).json({ message: 'Error creating user', error: err });
  }
});

// Authentication routes
app.post('/api/register', registerUser);
app.post('/api/login', loginUser);
app.post('/api/google-login', googleLogin);

// Protected test route example
app.get('/api/protected', authenticateToken, (req, res) => {
  res.json({ message: 'This is a protected route', user: req.user });
});

// Start the server
app.listen(port, () => {
  console.log(Node server running at http://localhost:${port});
});