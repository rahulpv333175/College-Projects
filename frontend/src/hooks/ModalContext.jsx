import React, { createContext, useContext, useState } from 'react';

// Create the context
const ModalContext = createContext();

// Create a custom hook to use the context
export const useModal = () => useContext(ModalContext);

// Create the provider component
export const ModalProvider = ({ children }) => {
  const [modal, setModal] = useState(null); // 'login', 'membership', 'signup', etc.
  const [modalProps, setModalProps] = useState({}); // To pass data, e.g., plan name

  // simple counter trigger to signal auth state changes
  const [authTrigger, setAuthTrigger] = useState(0);

  const openModal = (modalName, props = {}) => {
    setModal(modalName);
    setModalProps(props);
  };

  const closeModal = () => {
    setModal(null);
    setModalProps({});
  };

  // call this to notify consumers that auth changed (login/logout)
  const refreshAuth = () => setAuthTrigger((n) => n + 1);

  // The value that will be available to all children
  const value = {
    modal,
    openModal,
    closeModal,
    modalProps,
    // auth helpers
    authTrigger,
    refreshAuth,
  };

  return (
    <ModalContext.Provider value={value}>
      {children}
    </ModalContext.Provider>
  );
};
