import React from 'react';
import ClassCard from '../components/ClassCard'; // Import the new component
import { Search } from 'lucide-react';

// --- Hard-coded data for the UI ---
const classesData = [
  {
    title: "Ascend Strength",
    instructor: "Mike Thompson",
    time: "60 Mins",
    difficulty: "Advanced",
    image: "https://placehold.co/600x400/D4AF37/0B0B0F?text=Strength"
  },
  {
    title: "Ignite HIIT",
    instructor: "Sarah Chen",
    time: "45 Mins",
    difficulty: "Intermediate",
    image: "https://placehold.co/600x400/BFBFBF/0B0B0F?text=HIIT"
  },
  {
    title: "Flow Yoga",
    instructor: "Elena Rodriguez",
    time: "60 Mins",
    difficulty: "Beginner",
    image: "https://placehold.co/600x400/BFBFBF/0B0B0F?text=Yoga"
  },
  {
    title: "Rhythm Spin",
    instructor: "DJ Havoc",
    time: "50 Mins",
    difficulty: "Intermediate",
    image: "https://placehold.co/600x400/BFBFBF/0B0B0F?text=Spin"
  },
  {
    title: "Zen Meditation",
    instructor: "Elena Rodriguez",
    time: "30 Mins",
    difficulty: "Beginner",
    image: "https://placehold.co/600x400/BFBFBF/0B0B0F?text=Meditation"
  },
  {
    title: "Core Crusher",
    instructor: "Mike Thompson",
    time: "30 Mins",
    difficulty: "Intermediate",
    image: "https://placehold.co/600x400/BFBFBF/0B0B0F?text=Core"
  },
];

const ClassesPage = () => {
  return (
    <div className="container mx-auto px-6 py-20">
      <h1 className="text-5xl font-bold text-center mb-16">
        FIND YOUR <span className="text-accent">CLASS</span>
      </h1>

      {/* --- Filter Bar --- */}
      <div className="mb-12 p-6 bg-gray-800/50 rounded-xl border border-gray-700 grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
        
        {/* Search Input */}
        <div className="md:col-span-2">
          <label htmlFor="search" className="block text-sm font-medium text-gray-300 mb-2">Search by Name</label>
          <div className="relative">
            <input 
              type="text" 
              id="search" 
              className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 pl-10 text-white focus:border-accent focus:ring-accent focus:ring-1 outline-none" 
              placeholder="e.g., 'Ignite HIIT'..."
            />
            <Search size={20} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
          </div>
        </div>

        {/* Filter Dropdown */}
        <div>
          <label htmlFor="type" className="block text-sm font-medium text-gray-300 mb-2">Class Type</label>
          <select id="type" name="type" className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white focus:border-accent focus:ring-accent focus:ring-1 outline-none">
            <option>All Types</option>
            <option>Strength</option>
            <option>Cardio</option>
            <option>Yoga</option>
            <option>Spin</option>
          </select>
        </div>
        
        {/* Filter Button */}
        <button className="btn-primary h-[48px]">
          Apply Filters
        </button>
      </div>
      

      {/* --- Class Grid --- */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {classesData.map((classItem, index) => (
          <ClassCard 
            key={index}
            title={classItem.title}
            instructor={classItem.instructor}
            time={classItem.time}
            difficulty={classItem.difficulty}
            image={classItem.image}
          />
        ))}
      </div>
    </div>
  );
};

export default ClassesPage;