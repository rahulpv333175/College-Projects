import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { Menu, X } from 'lucide-react';
import { useModal } from '../hooks/ModalContext.jsx';
import { jwtDecode } from "jwt-decode";

const Navbar = () => {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [userName, setUserName] = useState("");

  const { openModal } = useModal();

  // Check login status on load + decode token
  useEffect(() => {
    const token = localStorage.getItem("token");

    if (token) {
      setIsLoggedIn(true);

      try {
        const decoded = jwtDecode(token);
        setUserName(decoded.name || decoded.email || "User");
      } catch (err) {
        console.error("TOKEN DECODE ERROR:", err);
      }

    } else {
      setIsLoggedIn(false);
      setUserName("");
    }
  }, []);

  const handleLoginClick = () => {
    setIsMobileMenuOpen(false);
    openModal('login');
  };

  const handleSignupClick = () => {
    setIsMobileMenuOpen(false);
    openModal('signup');
  };

  const handleLogout = () => {
    localStorage.removeItem("token");
    setIsLoggedIn(false);
    setUserName("");
    alert("Logged out!");
    setIsMobileMenuOpen(false);
  };

  return (
    <header className="sticky top-0 z-40 bg-dark-bg/80 backdrop-blur-lg border-b border-gray-800">
      <nav className="container mx-auto px-6 py-4 flex justify-between items-center">

        {/* Logo */}
        <Link to="/" className="text-3xl font-black tracking-tighter text-white">
          ASCEND<span className="text-accent">.</span>
        </Link>

        {/* Desktop Nav */}
        <div className="hidden md:flex space-x-8 items-center">
          <a href="/#features" className="text-gray-300 hover:text-accent transition-colors">Features</a>
          <a href="/#ai-tools" className="text-gray-300 hover:text-accent transition-colors">AI Tools</a>
          <Link to="/classes" className="text-gray-300 hover:text-accent transition-colors">Classes</Link>
          <a href="/#pricing" className="text-gray-300 hover:text-accent transition-colors">Pricing</a>
          <a href="/#contact" className="text-gray-300 hover:text-accent transition-colors">Contact</a>

          {/* Logged-in → Show My Account + Logout */}
          {isLoggedIn ? (
            <>
              <Link 
                to="/account"
                className="text-accent font-semibold hover:underline"
              >
                Hi, {userName}
              </Link>

              <button
                onClick={handleLogout}
                className="bg-red-600 text-white py-2 px-5 rounded-full hover:bg-red-500 transition-colors text-sm font-medium"
              >
                Log Out
              </button>
            </>
          ) : (
            <>
              <button
                onClick={handleLoginClick}
                className="bg-gray-800 text-white py-2 px-5 rounded-full hover:bg-gray-700 transition-colors text-sm font-medium"
              >
                Log In
              </button>

              <button
                onClick={handleSignupClick}
                className="bg-accent text-black py-2 px-5 rounded-full hover:bg-accent/80 transition-colors text-sm font-bold"
              >
                Sign Up
              </button>
            </>
          )}
        </div>

        {/* Mobile Menu Button */}
        <div className="md:hidden">
          <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}>
            {isMobileMenuOpen ? <X className="w-6 h-6 text-white" /> : <Menu className="w-6 h-6 text-white" />}
          </button>
        </div>
      </nav>

      {/* Mobile Menu */}
      {isMobileMenuOpen && (
        <div className="md:hidden bg-dark-bg/95 border-t border-gray-800 pb-4">

          <a href="/#features" className="block text-center text-gray-300 hover:bg-gray-800 py-3">Features</a>
          <a href="/#ai-tools" className="block text-center text-gray-300 hover:bg-gray-800 py-3">AI Tools</a>
          <Link to="/classes" className="block text-center text-gray-300 hover:bg-gray-800 py-3">Classes</Link>
          <a href="/#pricing" className="block text-center text-gray-300 hover:bg-gray-800 py-3">Pricing</a>
          <a href="/#contact" className="block text-center text-gray-300 hover:bg-gray-800 py-3">Contact</a>

          {/* Mobile Logged-In Version */}
          {isLoggedIn ? (
            <>
              <Link 
                to="/account"
                onClick={() => setIsMobileMenuOpen(false)}
                className="block text-center text-accent font-semibold py-3 hover:underline"
              >
                Hi, {userName}
              </Link>

              <button
                onClick={handleLogout}
                className="block w-[calc(100%-2rem)] text-center bg-red-600 text-white py-3 my-2 mx-4 rounded-full font-medium"
              >
                Log Out
              </button>
            </>
          ) : (
            <>
              <button
                onClick={handleLoginClick}
                className="block w-[calc(100%-2rem)] text-center bg-gray-800 text-white py-3 my-2 mx-4 rounded-full font-medium"
              >
                Log In
              </button>

              <button
                onClick={handleSignupClick}
                className="block w-[calc(100%-2rem)] text-center bg-accent text-black py-3 my-2 mx-4 rounded-full font-bold"
              >
                Sign Up
              </button>
            </>
          )}
        </div>
      )}
    </header>
  );
};

export default Navbar;
