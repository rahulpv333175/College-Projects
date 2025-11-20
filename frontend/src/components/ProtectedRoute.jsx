import React from "react";
import { Navigate } from "react-router-dom";

const ProtectedRoute = ({ children }) => {
  const token = localStorage.getItem("token");

  if (!token) {
    // Not logged in → redirect to homepage or login page
    alert("Please log in to continue.");
    return <Navigate to="/" replace />;
  }

  return children;
};

export default ProtectedRoute;
