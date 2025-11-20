import express from "express";
import multer from "multer";
import cloudinary from "../config/cloudinary.js";
import { CloudinaryStorage } from "multer-storage-cloudinary";

const router = express.Router();

// Multer + Cloudinary Storage config
const storage = new CloudinaryStorage({
  cloudinary,
  params: {
    folder: "ascend-gym-profiles",
    allowed_formats: ["jpg", "jpeg", "png"],
  },
});

const upload = multer({ storage });

// UPLOAD ENDPOINT (wrapped to capture multer/storage errors)
router.post("/profile", (req, res) => {
  // call multer single handler and catch errors in the callback
  upload.single("image")(req, res, (err) => {
    if (err) {
      console.error("MULTER/UPLOAD ERROR:", err);
      // If multer passed a message, include it
      const message = err.message || "Image upload failed";
      return res.status(500).json({ message });
    }

    // No file uploaded
    if (!req.file) {
      console.error("UPLOAD ERROR: no file received on request", {
        headers: req.headers && { ...req.headers },
      });
      return res.status(400).json({ message: "No file uploaded" });
    }

    // Success
    try {
      console.log("UPLOAD SUCCESS: file info", {
        originalname: req.file.originalname,
        mimetype: req.file.mimetype,
        size: req.file.size,
        path: req.file.path,
      });

      return res.json({
        message: "Upload successful",
        imageUrl: req.file.path, // Cloudinary URL
      });
    } catch (err2) {
      console.error("UPLOAD RESPONSE ERROR:", err2);
      return res.status(500).json({ message: "Image upload failed" });
    }
  });
});

export default router;
