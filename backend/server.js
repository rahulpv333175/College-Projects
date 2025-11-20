import express from 'express';
import cors from 'cors';
import 'dotenv/config'; // This MUST be one of the first imports
// DEBUG: Print Cloudinary env variables before importing Cloudinary
console.log('DEBUG ENV:', {
  CLOUDINARY_CLOUD_NAME: process.env.CLOUDINARY_CLOUD_NAME,
  CLOUDINARY_API_KEY: process.env.CLOUDINARY_API_KEY,
  CLOUDINARY_API_SECRET: process.env.CLOUDINARY_API_SECRET ? '***HIDDEN***' : undefined
});
import { GoogleGenerativeAI } from '@google/generative-ai';
import authRoutes from './routes/auth.routes.js';
import userRoutes from "./routes/user.routes.js";

import mongoose from 'mongoose';

mongoose
  .connect(process.env.MONGODB_URI, {
    dbName: "ascend-gym"
  })
  .then(() => console.log("MongoDB connected successfully"))
  .catch((err) => {
    console.error("MongoDB connection error:", err);
  });


// --- App Setup ---
const app = express();
const port = process.env.PORT || 5001;

// --- Middleware ---
app.use(cors());
app.use(express.json());
app.use('/api/auth', authRoutes);
app.use("/api/user", userRoutes);

// Request logger to help debug missing routes / auth headers
app.use((req, res, next) => {
  try {
    console.log('INCOMING:', req.method, req.path, 'Headers:', {
      authorization: req.headers.authorization,
      'content-type': req.headers['content-type'],
    });
  } catch (e) {
    // ignore logging errors
  }
  next();
});


// --- !! DEBUGGING !! ---
// We will try to set up the AI. If it fails, we'll know immediately.
let genAI;
let model;
try {
  // This line will print the key (or 'undefined') to your terminal
  console.log("DEBUG: Trying to load API Key:", process.env.GEMINI_API_KEY);

  if (!process.env.GEMINI_API_KEY) {
    throw new Error("GEMINI_API_KEY is not defined in your .env file!");
  }

  genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
  model = genAI.getGenerativeModel({ model: "gemini-2.5-flash-preview-09-2025" });
  
  console.log("DEBUG: Google AI SDK initialized successfully.");

} catch (error) {
  console.error("!!!!!!!!!!!!!!!!! FATAL ERROR INITIALIZING GOOGLE AI !!!!!!!!!!!!!!!!!!");
  console.error("This is likely an API Key or .env file problem.");
  console.error(error.message); // Print the specific error
  console.error("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
}
// --- END DEBUGGING ---


// --- API Routes ---

app.get('/api', (req, res) => {
  res.json({ message: 'Welcome to the Ascend Gym API!' });
});


// --- AI Chatbot Route ---
app.post('/api/chat', async (req, res) => {
  if (!model) {
    console.error("DEBUG: /api/chat called, but 'model' is not initialized. Check server start-up logs.");
    return res.status(500).json({ error: 'AI service is not configured correctly. Check server logs.' });
  }

  try {
    const { history, prompt } = req.body; 
    
    if (!prompt) {
      return res.status(400).json({ error: 'Prompt is required.' });
    }

    const chat = model.startChat({
        history: history,
        generationConfig: {
          maxOutputTokens: 500,
        },
        systemInstruction: {
             parts: [{ text: "You are 'Ascend AI', a helpful and motivating fitness and nutrition assistant for the Ascend gym. Be concise and friendly." }]
        },
    });

    const result = await chat.sendMessage(prompt);
    const response = result.response;
    const text = response.text();
    
    res.json({ text });

  } catch (error) {
    // --- !! NEW, LOUD, IMPOSSIBLE-TO-MISS LOGGING !! ---
    console.log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    console.log("---!!! THE /api/chat ROUTE CRASHED. THIS IS THE BACKEND TERMINAL !!!---");
    console.log("THE FULL ERROR FROM GOOGLE IS BELOW:");
    console.error(error); // This will print the *entire* error object
    console.log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
    
    // Send a new, unique error message to the frontend
    res.status(500).json({ 
        error: "This is a custom 500 error from the backend. The backend terminal has the full details.",
        details: error.message 
    });
  }
});

// --- AI Generator Route ---
app.post('/api/generate', async (req, res) => {
  if (!model) {
    console.error("DEBUG: /api/generate called, but 'model' is not initialized. Check server start-up logs.");
    return res.status(500).json({ error: 'AI service is not configured correctly. Check server logs.' });
  }

    try {
        const { systemPrompt } = req.body;
        if (!systemPrompt) {
            return res.status(400).json({ error: 'A system prompt is required.' });
        }
        
        const result = await model.generateContent(systemPrompt);
        const response = result.response;
        const text = response.text();

        res.json({ text });

    } catch (error) {
        // --- !! NEW, LOUD, IMPOSSIBLE-TO-MISS LOGGING !! ---
        console.log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");
        console.log("---!!! THE /api/generate ROUTE CRASHED. THIS IS THE BACKEND TERMINAL !!!---");
        console.log("THE FULL ERROR FROM GOOGLE IS BELOW:");
        console.error(error); // This will print the *entire* error object
        console.log("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!");

        // Send a new, unique error message to the frontend
        res.status(500).json({ 
            error: "This is a custom 500 error from the backend. The backend terminal has the full details.",
            details: error.message 
        });
    }
});


// --- Start Server (robust) ---
const MAX_PORT_ATTEMPTS = 5;
const startServer = (startPort, attempt = 0) => {
  try {
    const server = app.listen(startPort, () => {
      console.log(`Server running on http://localhost:${startPort}`);
    });

    server.on('error', (err) => {
      if (err.code === 'EADDRINUSE' && attempt < MAX_PORT_ATTEMPTS) {
        console.warn(`Port ${startPort} in use, trying port ${startPort + 1}...`);
        startServer(startPort + 1, attempt + 1);
      } else {
        console.error('Server error:', err);
        process.exit(1);
      }
    });

  } catch (err) {
    if (err.code === 'EADDRINUSE' && attempt < MAX_PORT_ATTEMPTS) {
      console.warn(`Port ${startPort} in use (caught), trying port ${startPort + 1}...`);
      startServer(startPort + 1, attempt + 1);
    } else {
      console.error('Failed to start server:', err);
      process.exit(1);
    }
  }
};

startServer(port);