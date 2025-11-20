import React from 'react';
import { useModal } from '../hooks/ModalContext.jsx';
import { X, CheckCircle } from 'lucide-react';

const MembershipModal = () => {
  const { modal, closeModal, modalProps } = useModal();
  const isOpen = modal === 'membership';

  if (!isOpen) return null;

  const { planName, price } = modalProps; // Get the plan name from context

  return (
    <>
      {/* Overlay */}
      <div 
        className="fixed inset-0 bg-black/70 z-50 transition-opacity duration-300"
        onClick={closeModal}
      ></div>

      {/* Modal Content */}
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 bg-gray-900 w-full max-w-md p-8 rounded-xl shadow-2xl z-50 border border-gray-700 text-center">
        <button 
          onClick={closeModal}
          className="absolute top-4 right-4 text-gray-400 hover:text-white"
        >
          <X size={24} />
        </button>
        
        <CheckCircle size={64} className="text-accent mx-auto mb-6" />

        <h2 className="text-3xl font-bold text-center text-white mb-4">
          Great Choice!
        </h2>
        <p className="text-lg text-gray-300 mb-6">
          You've selected the <span className="font-bold text-accent">{planName}</span> plan for <span className="font-bold text-white">${price}/mo</span>.
        </p>
        <p className="text-gray-400 mb-8">
          This is a demo. In a real app, this would lead to a checkout page.
        </p>

        <button onClick={closeModal} className="btn-primary w-full">
          Confirm & Proceed
        </button>
      </div>
    </>
  );
};

export default MembershipModal;