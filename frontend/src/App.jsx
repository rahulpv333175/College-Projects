import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import MainLayout from './components/MainLayout';
import HomePage from './pages/HomePage';
import ClassesPage from './pages/ClassesPage';
import LoginPage from './pages/LoginPage';
import MyAccount from './pages/MyAccount';


// Modals
import LoginModal from './components/LoginModal';
import MembershipModal from './components/MembershipModal';

// ProtectedRoute
import ProtectedRoute from './components/ProtectedRoute';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<HomePage />} />

          {/* Protected Classes Route */}
          <Route 
            path="classes" 
            element={
              <ProtectedRoute>
                <ClassesPage />
              </ProtectedRoute>
            } 
          />
          <Route 
            path="account"
            element={
              <ProtectedRoute>
                <MyAccount />
              </ProtectedRoute>
            }
          />


          {/* Add more protected/unprotected pages here */}
        </Route>

        {/* Optional standalone login page */}
        <Route path="/login" element={<LoginPage />} />
      </Routes>

      {/* Global Modals */}
      <LoginModal />
      <MembershipModal />
    </BrowserRouter>
  );
}

export default App;
