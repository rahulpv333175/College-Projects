import React from 'react';
import { Clock, User, BarChart } from 'lucide-react';

/**
 * A reusable card component to display a single class.
 * Includes a sleek hover effect.
 */
const ClassCard = ({ title, instructor, time, difficulty, image }) => {
  return (
    <div className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700 transition-all duration-300 hover:border-accent hover:shadow-2xl hover:-translate-y-2 group">
      <div className="relative h-48 w-full overflow-hidden">
        <img 
          src={image} 
          alt={title} 
          className="w-full h-full object-cover transition-transform duration-300 group-hover:scale-110" 
        />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent"></div>
        <span className="absolute top-3 right-3 bg-accent/80 text-gray-900 text-xs font-bold py-1 px-3 rounded-full uppercase tracking-wider">
          {difficulty}
        </span>
      </div>
      <div className="p-6">
        <h3 className="text-2xl font-bold text-white mb-3">{title}</h3>
        <div className="space-y-3 text-gray-400">
          <div className="flex items-center gap-2">
            <User size={16} className="text-accent" />
            <span>{instructor}</span>
          </div>
          <div className="flex items-center gap-2">
            <Clock size={16} className="text-accent" />
            <span>{time}</span>
          </div>
        </div>
        <button className="btn-secondary w-full mt-6 text-sm py-2">
          View Details
        </button>
      </div>
    </div>
  );
};

export default ClassCard;