import express from "express";
import jwt from "jsonwebtoken";
import User from "../models/User.model.js";

const router = express.Router();

const JWT_SECRET = process.env.JWT_SECRET;

// 🔐 Middleware — verify user token
function authMiddleware(req, res, next) {
  const authHeader = req.headers.authorization;

  if (!authHeader) {
    console.warn('AUTH: no Authorization header on request to', req.path);
    return res.status(401).json({ message: "No token provided" });
  }

  const token = authHeader.split(" ")[1];

  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded; // attach user data
    console.log('AUTH: token decoded for user', decoded.id || decoded._id || decoded.email);
    next();
  } catch (err) {
    console.warn('AUTH: invalid token for request to', req.path, err.message);
    return res.status(401).json({ message: "Invalid token" });
  }
}

/* -----------------------------------------
  UPDATE PROFILE (image + membership)
*/
router.put("/update", authMiddleware, async (req, res) => {
  try {
    const { profileImage, membershipPlan } = req.body;

    console.log('USER UPDATE: called by user id:', req.user && req.user.id, 'body:', req.body);

    const user = await User.findById(req.user.id);

    if (!user) return res.status(404).json({ message: "User not found" });

    if (profileImage) user.profileImage = profileImage;
    if (membershipPlan) user.membershipPlan = membershipPlan;

    await user.save();

    res.json({
      message: "Profile updated successfully",
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        profileImage: user.profileImage,
        membershipPlan: user.membershipPlan,
      }
    });

  } catch (err) {
    console.error("PROFILE UPDATE ERROR:", err);
    res.status(500).json({ message: "Server error while updating profile" });
  }
});
/* -----------------------------------------
   GET CURRENT USER PROFILE
------------------------------------------ */
router.get("/me", authMiddleware, async (req, res) => {
  try {
    const user = await User.findById(req.user.id);
    if (!user) return res.status(404).json({ message: "User not found" });
    res.json({ user: {
      id: user._id,
      name: user.name,
      email: user.email,
      profileImage: user.profileImage,
      membershipPlan: user.membershipPlan,
      iat: req.user.iat // from JWT
    }});
  } catch (err) {
    console.error("FETCH USER ERROR:", err);
    res.status(500).json({ message: "Server error while fetching user" });
  }
});

export default router;
