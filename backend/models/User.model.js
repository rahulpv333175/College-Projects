import mongoose from 'mongoose';
import bcrypt from 'bcryptjs';

const userSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: true,
      trim: true
    },

    email: {
      type: String,
      required: true,
      unique: true,
      lowercase: true,
      trim: true
    },

    password: {
      type: String,
      required: true,
      minlength: 3
    },

    // 🔥 Profile Image — NEW FIELD
    profileImage: {
      type: String,
      default: "https://i.ibb.co/YWs4kC0/default-avatar.png"
    },

    // 🔥 Membership Plan — NEW FIELD
    membershipPlan: {
      type: String,
      enum: ["free", "basic", "premium"],
      default: "free"
    }
  },
  {
    timestamps: true
  }
);

// Auto hash password if changed
userSchema.pre('save', async function (next) {
  if (!this.isModified('password')) return next();
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});

const User = mongoose.model('User', userSchema);
export default User;
