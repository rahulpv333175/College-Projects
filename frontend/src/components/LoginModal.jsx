import React, { useState } from 'react';
import axios from "axios";
import { useModal } from '../hooks/ModalContext.jsx';
import { X } from 'lucide-react';

const LoginModal = () => {
  const { modal, closeModal, openModal, refreshAuth } = useModal();
  const isOpen = modal === 'login';

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();

    console.log("LOGIN DATA:", { email, password });

    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5002/api/auth/login", {
        email,
        password,
      });

      console.log("LOGIN SUCCESS:", res.data);

      // Save token
      localStorage.setItem("token", res.data.token);

      // 🔥 Tell Navbar to update immediately (NO page refresh needed)
      refreshAuth();

      alert("Login successful!");
      closeModal();

    } catch (err) {
      console.error("LOGIN ERROR:", err.response?.data || err);
      alert(err.response?.data?.message || "Invalid credentials");
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Overlay */}
      <div 
        className="fixed inset-0 bg-black/70 z-50 transition-opacity duration-300"
        onClick={closeModal}
      ></div>

      {/* Modal Content */}
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-gray-900 w-full max-w-md p-8 rounded-xl shadow-2xl z-50 border border-gray-700">

        {/* Close Button */}
        <button 
          onClick={closeModal}
          className="absolute top-4 right-4 text-gray-400 hover:text-white"
          aria-label="Close login"
        >
          <X size={24} />
        </button>

        <h2 className="text-3xl font-bold text-center text-white mb-2">
          Welcome <span className="text-accent">Back</span>
        </h2>
        <p className="text-center text-gray-400 mb-8">Log in to access your account.</p>

        <form className="space-y-6" onSubmit={handleSubmit}>

          {/* Email */}
          <div>
            <label htmlFor="email" className="block text-sm font-medium text-gray-300 mb-2">
              Email
            </label>
            <input 
              type="email" 
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white focus:border-accent focus:ring-accent focus:ring-1 outline-none" 
              placeholder="you@example.com" 
              required
            />
          </div>

          {/* Password */}
          <div>
            <label htmlFor="password" className="block text-sm font-medium text-gray-300 mb-2">
              Password
            </label>
            <input 
              type="password" 
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white focus:border-accent focus:ring-accent focus:ring-1 outline-none" 
              placeholder="••••••••" 
              required
            />
          </div>

          {/* Submit */}
          <button type="submit" className="btn-primary w-full" disabled={loading}>
            {loading ? "Logging in..." : "Log In"}
          </button>

          {/* Switch to Signup */}
          <p className="text-sm text-center text-gray-400">
            Don't have an account?{" "}
            <span
              onClick={() => {
                closeModal();
                setTimeout(() => openModal("signup"), 200);
              }}
              className="font-medium text-accent hover:underline cursor-pointer"
            >
              Sign up
            </span>
          </p>

        </form>
      </div>
    </>
  );
};

export default LoginModal;
