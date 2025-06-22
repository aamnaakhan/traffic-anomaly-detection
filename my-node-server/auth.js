const jwt = require('jsonwebtoken');
const bcrypt = require('bcrypt');
const User = require('./models/user');

const JWT_SECRET = 'your_jwt_secret_key'; // Replace with env variable in production

// Generate JWT token
function generateToken(user) {
  return jwt.sign({ id: user._id, email: user.email }, JWT_SECRET, { expiresIn: '1h' });
}

// Middleware to verify JWT token
function authenticateToken(req, res, next) {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];
  if (!token) return res.status(401).json({ message: 'Access token missing' });

  jwt.verify(token, JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ message: 'Invalid token' });
    req.user = user;
    next();
  });
}

// Register new user (for testing/demo)
async function registerUser(req, res) {
  let { name, email, password } = req.body;
  if (!email || !password) return res.status(400).json({ message: 'Missing fields' });
  if (!name) name = '';

  const existingUser = await User.findOne({ email });
  if (existingUser) return res.status(400).json({ message: 'User already exists' });

  const hashedPassword = await bcrypt.hash(password, 10);
  const newUser = new User({ name, email, password: hashedPassword });
  await newUser.save();

  const token = generateToken(newUser);
  res.status(201).json({ token, user: { id: newUser._id, name, email } });
}

// Login user
async function loginUser(req, res) {
  const { email, password } = req.body;
  if (!email || !password) return res.status(400).json({ message: 'Missing email or password' });

  const user = await User.findOne({ email });
  if (!user) return res.status(400).json({ message: 'Invalid credentials' });

  const validPassword = await bcrypt.compare(password, user.password);
  if (!validPassword) return res.status(400).json({ message: 'Invalid credentials' });

  const token = generateToken(user);
  res.json({ token, user: { id: user._id, name: user.name, email } });
}

// Mock Google login (for demo)
async function googleLogin(req, res) {
  const { tokenId } = req.body;
  // In real app, verify tokenId with Google API
  if (!tokenId) return res.status(400).json({ message: 'Missing tokenId' });

  // Mock user info from token
  const email = 'googleuser@example.com';
  let user = await User.findOne({ email });
  if (!user) {
    user = new User({ name: 'Google User', email, password: '' });
    await user.save();
  }

  const token = generateToken(user);
  res.json({ token, user: { id: user._id, name: user.name, email } });
}

module.exports = {
  authenticateToken,
  registerUser,
  loginUser,
  googleLogin,
};
