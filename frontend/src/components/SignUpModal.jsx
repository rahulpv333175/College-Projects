import React, { useState } from "react";
import axios from "axios";
import { useModal } from "../hooks/ModalContext.jsx";
import { X } from "lucide-react";

const SignUpModal = () => {
  const { modal, closeModal, openModal } = useModal();
  const isOpen = modal === "signup";

  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Debug: show what will be sent
    console.log("SIGNUP DATA:", { name, email, password });

    setLoading(true);
    try {
      const res = await axios.post("http://localhost:5002/api/auth/register", {
        name,
        email,
        password,
      });

      console.log("REGISTER SUCCESS:", res.data);

      // Save token
      localStorage.setItem("token", res.data.token);

      alert("Signup successful!");
      closeModal();

      // Optionally open login (or keep logged in). We'll keep the user logged in by default.
      // If you prefer to show login modal instead, uncomment next lines:
      // setTimeout(() => openModal("login"), 200);

    } catch (err) {
      console.error("REGISTER ERROR:", err.response?.data || err);
      alert(err.response?.data?.message || "Signup failed");
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
        <button
          onClick={closeModal}
          className="absolute top-4 right-4 text-gray-400 hover:text-white"
          aria-label="Close sign up"
        >
          <X size={24} />
        </button>

        <h2 className="text-3xl font-bold text-center text-white mb-2">
          Create <span className="text-accent">Account</span>
        </h2>
        <p className="text-center text-gray-400 mb-8">Join the Ascend family!</p>

        <form className="space-y-6" onSubmit={handleSubmit}>
          {/* Name */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Full Name
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white focus:border-accent focus:ring-accent focus:ring-1 outline-none"
              placeholder="Rahul Sharma"
              required
            />
          </div>

          {/* Email */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white focus:border-accent focus:ring-accent focus:ring-1 outline-none"
              placeholder="you@example.com"
              required
            />
          </div>

          {/* Password */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Password
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white focus:border-accent focus:ring-accent focus:ring-1 outline-none"
              placeholder="••••••••"
              required
            />
          </div>

          <button type="submit" className="btn-primary w-full" disabled={loading}>
            {loading ? "Signing up..." : "Sign Up"}
          </button>

          <p className="text-sm text-center text-gray-400">
            Already have an account?{" "}
            <span
              onClick={() => {
                closeModal();
                setTimeout(() => openModal("login"), 200);
              }}
              className="font-medium text-accent hover:underline cursor-pointer"
            >
              Log in
            </span>
          </p>
        </form>
      </div>
    </>
  );
};

export default SignUpModal;
