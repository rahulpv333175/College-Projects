import React from 'react';
import { useModal } from '../hooks/ModalContext.jsx';
import { Check, X } from 'lucide-react';

const PricingCard = ({ plan, price, features, featured = false }) => {
  const { openModal } = useModal();

  const handleChoosePlan = () => {
    // Pass the plan name to the modal
    openModal('membership', { planName: plan, price: price });
  };

  return (
    <div className={`p-8 rounded-xl border ${featured ? 'bg-gray-800 border-accent scale-105' : 'bg-gray-800/50 border-gray-700 transition-transform duration-300 hover:scale-105'}`}>
      <h3 className="text-2xl font-bold text-center mb-2">{plan}</h3>
      <p className="text-5xl font-extrabold text-center mb-6">
        ${price}<span className="text-lg font-normal text-gray-400">/mo</span>
      </p>
      <ul className="space-y-3 text-left mb-8">
        {features.map((feat, i) => (
          <li key={i} className="flex items-center">
            {feat.included ? (
              <Check className="w-5 h-5 mr-3 text-accent flex-shrink-0" />
            ) : (
              <X className="w-5 h-5 mr-3 text-gray-500 flex-shrink-0" />
            )}
            <span className={!feat.included ? 'text-gray-500 line-through' : 'text-gray-300'}>
              {feat.text}
            </span>
          </li>
        ))}
      </ul>
      <button 
        onClick={handleChoosePlan}
        className={`w-full ${featured ? 'btn-primary' : 'btn-secondary'}`}
      >
        Choose Plan
      </button>
    </div>
  );
};

export default PricingCard;