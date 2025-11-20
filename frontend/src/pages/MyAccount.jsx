import React, { useEffect, useState } from "react";
import axios from "axios";

const MyAccount = () => {
  const [user, setUser] = useState(null);
  const [showPresets, setShowPresets] = useState(false);

  useEffect(() => {
    const fetchUser = async () => {
      const token = localStorage.getItem("token");
      if (!token) return;
      try {
        const res = await axios.get("http://localhost:5002/api/user/me", {
          headers: { Authorization: `Bearer ${token}` }
        });
        setUser(res.data.user);
      } catch (err) {
        console.error("FETCH USER ERROR:", err);
      }
    };
    fetchUser();
  }, []);

  // Preset avatars list (you can change these URLs)
  const presetAvatars = [
    "https://i.ibb.co/YWs4kC0/default-avatar.png",
    "https://res.cloudinary.com/darsswqbk/image/upload/v1763567173/ascend-gym-profiles/my67nnwyz2vzlyu96poi.png",
    "https://res.cloudinary.com/darsswqbk/image/upload/v1763567530/ascend-gym-profiles/kis2bkog9lptepwzmxyh.png",
    "https://i.ibb.co/7QpKsCX/face-1.png",
    "https://i.ibb.co/2y2G3z1/face-2.png",
  ];

  const handleSelectPreset = async (imageUrl) => {
    // optimistic UI update
    setUser((prev) => ({ ...prev, profileImage: imageUrl }));

    const token = localStorage.getItem("token");
    if (!token) {
      alert("Selected avatar set locally. Log in to persist this change.");
      setShowPresets(false);
      return;
    }

    try {
      const res = await axios.put(
        "http://localhost:5002/api/user/update",
        { profileImage: imageUrl },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      setUser(res.data.user);
      alert("Profile picture updated!");
    } catch (err) {
      console.error("PRESET UPDATE ERROR:", err);
      const serverMessage = err?.response?.data?.message || err?.message;
      alert(`Failed to persist avatar: ${serverMessage}`);
    } finally {
      setShowPresets(false);
    }
  };

  if (!user) {
    return (
      <div className="text-center text-white mt-20 text-2xl">
        Loading profile...
      </div>
    );
  }

  const handleUpgrade = async (plan) => {
    try {
      const token = localStorage.getItem("token");

      const res = await axios.put(
        "http://localhost:5002/api/user/update",
        { membershipPlan: plan },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      // Update UI with backend response
      setUser(res.data.user);
      alert(`Upgraded to ${plan.toUpperCase()} successfully!`);

    } catch (err) {
      console.error("UPGRADE ERROR:", err);
      alert("Failed to upgrade membership.");
    }
  };

  return (
    <div className="max-w-3xl mx-auto py-16 px-6 text-white">
      <h1 className="text-4xl font-bold mb-10">
        My <span className="text-accent">Account</span>
      </h1>

      <div className="bg-gray-900 p-8 rounded-2xl border border-gray-700 shadow-xl flex items-center gap-8">
        
        {/* Profile Image */}
        <div className="relative group cursor-pointer">
          <img
            src={user.profileImage || "https://i.ibb.co/YWs4kC0/default-avatar.png"}
            alt="User Avatar"
            className="w-28 h-28 rounded-full border-2 border-accent shadow-lg object-cover"
          />

          {/* Avatar chooser */}
            <div className="absolute bottom-0 right-0">
              <button
                onClick={() => setShowPresets((s) => !s)}
                className="bg-accent text-black px-3 py-1 rounded-lg text-sm font-bold"
              >
                Choose Avatar
              </button>
            </div>
            {showPresets && (
              <div className="absolute z-10 top-full mt-3 right-0 bg-gray-800 p-3 rounded-lg shadow-lg w-64">
                <p className="text-sm text-gray-300 mb-2">Select a preset avatar</p>
                <div className="grid grid-cols-3 gap-2">
                  {presetAvatars.map((src) => (
                    <button
                      key={src}
                      onClick={() => handleSelectPreset(src)}
                      className="w-16 h-16 p-0 rounded overflow-hidden border-2 border-transparent hover:border-accent"
                      title="Select avatar"
                    >
                      <img src={src} alt="avatar" className="w-full h-full object-cover" />
                    </button>
                  ))}
                </div>
              </div>
            )}
        </div>

        {/* User Details */}
        <div className="space-y-3">
          <h2 className="text-3xl font-bold">
            {user.name}
          </h2>

          <p className="text-gray-300 text-lg">
            <span className="text-accent font-semibold">Email:</span> {user.email}
          </p>

          <p className="text-gray-300">
            <span className="text-accent font-semibold">User ID:</span> {user.id}
          </p>

          <p className="text-gray-300">
            <span className="text-accent font-semibold">Member Since:</span>{" "}
            {new Date(user.iat * 1000).toLocaleDateString()}
          </p>

          <p className="text-gray-300">
            <span className="text-accent font-semibold">Membership:</span>{" "}
            {user.membershipPlan || "free"}
          </p>
            {/* Upgrade Membership Section */}
            <div className="mt-6 space-y-4">
              <h3 className="text-xl font-bold text-white">Upgrade Membership</h3>
              {/* BASIC PLAN */}
              <button
                onClick={() => handleUpgrade("basic")}
                className="w-full bg-gray-800 hover:bg-gray-700 text-white py-3 rounded-lg font-semibold border border-gray-700"
              >
                Upgrade to BASIC — ₹299 / month
              </button>
              {/* PREMIUM PLAN */}
              <button
                onClick={() => handleUpgrade("premium")}
                className="w-full bg-accent text-black hover:bg-accent/80 py-3 rounded-lg font-bold"
              >
                Upgrade to PREMIUM — ₹599 / month
              </button>
            </div>
        </div>
      </div>

      <p className="text-gray-500 mt-5 text-sm">
        This information is securely decoded from your JWT token.
      </p>
    </div>
  );
};

export default MyAccount;
