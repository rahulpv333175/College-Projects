import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'
import { ModalProvider } from './hooks/ModalContext.jsx' // Import the provider

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    {/* Wrap the entire app in the ModalProvider */}
    <ModalProvider>
      <App />
    </ModalProvider>
  </React.StrictMode>,
)