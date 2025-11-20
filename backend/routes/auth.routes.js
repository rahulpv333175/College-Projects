// backend/routes/auth.routes.js
import express from 'express';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import User from '../models/User.model.js';

const router = express.Router();

const JWT_SECRET = process.env.JWT_SECRET || 'please-change-this-secret';

// Helper to create token
function createToken(user) {
  return jwt.sign(
    {
      id: user._id,
      email: user.email,
      name: user.name || ''
    },
    JWT_SECRET,
    { expiresIn: '7d' }
  );
}

// ========================
// REGISTER ROUTE
// ========================
router.post('/register', async (req, res) => {
  try {
    const { name, email, password } = req.body || {};

    console.log("REGISTER ATTEMPT:", { name, email, password });

    if (!name || !email || !password) {
      console.log("REGISTER ERROR: Missing fields");
      return res.status(400).json({ message: 'Name, email and password are required.' });
    }

    const existing = await User.findOne({ email: email.toLowerCase().trim() });

    console.log("EXISTING USER CHECK:", existing);

    if (existing) {
      return res.status(409).json({ message: 'Email already in use.' });
    }

    const salt = await bcrypt.genSalt(10);
    const hashed = await bcrypt.hash(password, salt);

    const newUser = new User({
      name: name.trim(),
      email: email.toLowerCase().trim(),
      password: hashed
    });

    await newUser.save();

    console.log("REGISTERED USER:", newUser);

    const token = createToken(newUser);

    res.status(201).json({
      token,
      user: { id: newUser._id, name: newUser.name, email: newUser.email }
    });
  } catch (err) {
    console.error('REGISTER ERROR:', err);
    res.status(500).json({ message: 'Server error during registration.' });
  }
});

// ========================
// LOGIN ROUTE
// ========================
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body || {};

    console.log("LOGIN ATTEMPT:", email, password);

    if (!email || !password) {
      console.log("LOGIN ERROR: missing email or password");
      return res.status(400).json({ message: 'Email and password are required.' });
    }

    const cleanEmail = email.toLowerCase().trim();

    console.log("CLEAN EMAIL:", cleanEmail);

    const user = await User.findOne({ email: cleanEmail });

    console.log("FOUND USER:", user);

    if (!user) {
      console.log("LOGIN FAILURE: user not found");
      return res.status(401).json({ message: 'Invalid credentials.' });
    }

    const isMatch = await bcrypt.compare(password, user.password);

    console.log("PASSWORD MATCH RESULT:", isMatch);

    if (!isMatch) {
      console.log("LOGIN FAILURE: password mismatch");
      return res.status(401).json({ message: 'Invalid credentials.' });
    }

    const token = createToken(user);

    console.log("LOGIN SUCCESS FOR:", user.email);

    res.json({
      token,
      user: { id: user._id, name: user.name, email: user.email }
    });

  } catch (err) {
    console.error('LOGIN ERROR:', err);
    res.status(500).json({ message: 'Server error during login.' });
  }
});

export default router;
