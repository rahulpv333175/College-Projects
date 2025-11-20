import React from 'react';
import { useModal } from '../hooks/ModalContext.jsx';
import { Link } from 'react-router-dom';

const LoginPage = () => {
  const { openModal } = useModal();

  return (
    <div className="container mx-auto px-6 py-24 text-center">
      <h1 className="text-5xl font-bold text-center mb-8">
        Login Page
      </h1>
      <p className="text-xl text-gray-400 mb-8">
        This is a standalone login page, but we are using a modal for a better experience.
      </p>
      <button 
        onClick={() => openModal('login')}
        className="btn-primary"
      >
        Open Login Modal
      </button>
      <p className="mt-8">
        <Link to="/" className="text-accent hover:underline">Back to Home</Link>
      </p>
    </div>
  );
};

export default LoginPage;