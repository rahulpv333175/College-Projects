import React from 'react';
// Import Link from react-router-dom
import { Link } from 'react-router-dom';
import { Facebook, Twitter, Instagram } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-gray-900 border-t border-gray-800 pt-16 pb-8">
      <div className="container mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          {/* Column 1: Brand */}
          <div>
            <h3 className="text-3xl font-black tracking-tighter text-white">
              ASCEND<span className="text-accent">.</span>
            </h3>
            <p className="text-gray-400 mt-4">
              Push your limits. Ascend to your peak.
            </p>
            <div className="flex space-x-4 mt-6">
              {/* External links should stay as <a> tags */}
              <a href="https://facebook.com" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-accent"><Facebook /></a>
              <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-accent"><Twitter /></a>
              <a href="https://instagram.com" target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-accent"><Instagram /></a>
            </div>
          </div>

          {/* Column 2: Quick Links - CHANGED TO <Link> and <a> */}
          <div>
            <h4 className="text-white font-bold text-lg mb-4">Quick Links</h4>
            <ul className="space-y-2">
              {/* On-page anchors should be <a> */}
              <li><a href="/#features" className="text-gray-400 hover:text-accent">Features</a></li>
              <li><a href="/#pricing" className="text-gray-400 hover:text-accent">Pricing</a></li>
              <li><a href="/#contact" className="text-gray-400 hover:text-accent">Contact Us</a></li>
              {/* Internal page links should be <Link> */}
              <li><Link to="/login" className="text-gray-400 hover:text-accent">Login</Link></li>
            </ul>
          </div>

          {/* Column 3: Classes - CHANGED TO <Link> */}
          <div>
            <h4 className="text-white font-bold text-lg mb-4">Classes</h4>
            <ul className="space-y-2">
              {/* Internal page links should be <Link> */}
              <li><Link to="/classes" className="text-gray-400 hover:text-accent">Strength & Conditioning</Link></li>
              <li><Link to="/classes" className="text-gray-400 hover:text-accent">Cardio & HIIT</Link></li>
              <li><Link to="/classes" className="text-gray-400 hover:text-accent">Yoga & Mind</Link></li>
              <li><Link to="/classes" className="text-gray-400 hover:text-accent">Boxing</Link></li>
            </ul>
          </div>

          {/* Column 4: Hours */}
          <div>
            <h4 className="text-white font-bold text-lg mb-4">Opening Hours</h4>
            <ul className="space-y-2 text-gray-400">
              <li>Mon - Fri: 5:00 AM - 11:00 PM</li>
              <li>Sat: 7:00 AM - 10:00 PM</li>
              <li>Sun: 8:00 AM - 8:00 PM</li>
            </ul>
          </div>
        </div>

        <div className="mt-16 pt-8 border-t border-gray-800 text-center text-gray-500">
          <p>&copy; {new Date().getFullYear()} Ascend Gym. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;