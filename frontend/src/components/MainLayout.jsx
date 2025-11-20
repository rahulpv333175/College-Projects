import React from 'react';
import { Outlet } from 'react-router-dom';
import Navbar from './Navbar';
import Footer from './Footer';
import Chatbot from './Chatbot';
import LoginModal from "./LoginModal";
import SignUpModal from "./SignUpModal";

/**
 * MainLayout wraps all standard pages.
 * It provides the consistent Navbar, Footer, and AI Chatbot.
 */
const MainLayout = () => {
  return (
    <div className="flex flex-col min-h-screen">
      {/* Navbar */}
      <Navbar />

      {/* Page Content */}
      <main className="flex-1">
        <Outlet />
      </main>

      {/* Footer */}
      <Footer />

      {/* AI Chatbot */}
      <Chatbot />

      {/* Global Modals */}
      <LoginModal />
      <SignUpModal />
    </div>
  );
};

export default MainLayout;
