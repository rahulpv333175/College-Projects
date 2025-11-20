import React, { useState, useEffect } from 'react';
import { Check, Dumbbell, Zap, Brain, HeartPulse, User, Bot, Loader } from 'lucide-react';
import axios from 'axios';
import PricingCard from '/src/components/PricingCard.jsx'; // Import the refactored component

const HomePage = () => {
  // This hook is for the scroll animations
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add('fade-in-visible');
          }
        });
      },
      {
        threshold: 0.1,
      }
    );

    const elements = document.querySelectorAll('.fade-in');
    elements.forEach((el) => observer.observe(el));

    // Cleanup
    return () => elements.forEach((el) => observer.unobserve(el));
  }, []);


  return (
    <div className="overflow-x-hidden">
      {/* Add this CSS for the animations */}
      <style>{`
        .fade-in {
          opacity: 0;
          transform: translateY(30px);
          transition: opacity 0.6s ease-out, transform 0.6s ease-out;
        }
        .fade-in-visible {
          opacity: 1;
          transform: translateY(0);
        }
        .hero-bg {
          animation: slowZoom 20s infinite alternate ease-in-out;
        }
        @keyframes slowZoom {
          from { transform: scale(1); }
          to { transform: scale(1.1); }
        }
      `}</style>

      <HeroSection />
      <FeaturesSection />
      <AiToolsSection />
      <PricingSection />
      <ContactSection />
    </div>
  );
};

// --- Hero Section ---
const HeroSection = () => {
  return (
    <section 
      id="home" 
      className="relative h-screen flex items-center justify-center text-center overflow-hidden"
    >
      {/* Background Image */}
      <div 
        className="absolute inset-0 z-0 hero-bg"
        style={{ 
          backgroundImage: `url(https://placehold.co/1920x1080/0B0B0F/111111?text=Hero+Image)`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
      >
        {/* Overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-dark-bg via-dark-bg/70 to-transparent"></div>
      </div>
      
      {/* Content */}
      <div className="relative z-10 p-6 flex flex-col items-center">
        <h1 className="text-5xl md:text-7xl lg:text-8xl font-black uppercase tracking-tighter text-white drop-shadow-lg animate-fade-in-down">
          PUSH YOUR <span className="text-accent">LIMITS</span>
        </h1>
        <p className="text-lg md:text-2xl text-gray-300 mt-6 max-w-3xl drop-shadow-md animate-fade-in-up">
          Welcome to Ascend. The ultimate fitness experience with state-of-the-art equipment, elite trainers, and AI-powered personalization.
        </p>
        <div className="mt-10 flex flex-col sm:flex-row gap-4 animate-fade-in-up">
          <a href="#pricing" className="btn-primary">
            Join Now
          </a>
          <a href="#features" className="btn-secondary">
            Learn More
          </a>
        </div>
      </div>
    </section>
  );
};

// --- Features Section ---
const FeaturesSection = () => (
  <section id="features" className="py-24 bg-gray-900">
    <div className="container mx-auto px-6">
      <h2 className="text-4xl md:text-5xl font-bold text-center mb-16 fade-in">
        WHY <span className="text-accent">ASCEND?</span>
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
        <FeatureCard 
          icon={<Dumbbell size={40} />} 
          title="Elite Equipment"
          description="Top-tier strength and cardio machines from the best brands in the world."
          className="fade-in"
        />
        <FeatureCard 
          icon={<Zap size={40} />} 
          title="Group Classes"
          description="High-energy HIIT, boxing, spin, and more to keep you motivated."
          className="fade-in"
        />
        <FeatureCard 
          icon={<Brain size={40} />} 
          title="Expert Trainers"
          description="Certified professionals dedicated to helping you achieve your goals."
          className="fade-in"
        />
        <FeatureCard 
          icon={<HeartPulse size={40} />} 
          title="Wellness & Recovery"
          description="Saunas, cold plunges, and therapy zones to optimize your recovery."
          className="fade-in"
        />
      </div>
    </div>
  </section>
);

const FeatureCard = ({ icon, title, description, className = '' }) => (
  <div className={`bg-gray-800 p-8 rounded-xl border border-gray-700 text-center transition-all duration-300 hover:border-accent hover:-translate-y-2 ${className}`}>
    <div className="text-accent inline-block mb-6">{icon}</div>
    <h3 className="text-2xl font-bold mb-4">{title}</h3>
    <p className="text-gray-400">{description}</p>
  </div>
);


// --- AI Tools Section ---
const AiToolsSection = () => {
  const [workoutResult, setWorkoutResult] = useState('');
  const [workoutLoading, setWorkoutLoading] = useState(false);
  const [nutritionResult, setNutritionResult] = useState('');
  const [nutritionLoading, setNutritionLoading] = useState(false);

  // This function calls your backend
  const getAiResponse = async (systemPrompt) => {
    // Calling '/api/generate' which Vite proxies to your backend
    const response = await axios.post('/api/generate', { systemPrompt });
    return response.data.text;
  };

  const handleWorkoutSubmit = async (e) => {
    e.preventDefault();
    setWorkoutLoading(true);
    setWorkoutResult('');
    const formData = new FormData(e.target);
    const prompt = `Generate a detailed, day-1 workout plan for a user with the following goals:
- Goal: ${formData.get('goal')}
- Available Time: ${formData.get('time')}
- Equipment: ${formData.get('equipment')}
Format the response clearly with exercises, sets, and reps.`;
    
    try {
      const result = await getAiResponse(prompt);
      setWorkoutResult(result);
    } catch (error) {
      console.error(error);
      setWorkoutResult('Sorry, an error occurred while generating your workout.');
    } finally {
      setWorkoutLoading(false);
    }
  };
  
  const handleNutritionSubmit = async (e) => {
    e.preventDefault();
    setNutritionLoading(true);
    setNutritionResult('');
    const formData = new FormData(e.target);
    const prompt = `Generate a sample one-day meal plan for a user with the following goals:
- Goal: ${formData.get('nutrition-goal')}
- Dietary Preference: ${formData.get('preference')}
Format the response clearly with Breakfast, Lunch, Dinner, and Snacks.`;
    
    try {
      const result = await getAiResponse(prompt);
      setNutritionResult(result);
    } catch (error) {
      console.error(error);
      setNutritionResult('Sorry, an error occurred while generating your meal plan.');
    } finally {
      setNutritionLoading(false);
    }
  };

  return (
    <section id="ai-tools" className="py-24">
      <div className="container mx-auto px-6">
        <h2 className="text-4xl md:text-5xl font-bold text-center mb-16 fade-in">
          YOUR <span className="text-accent">AI ASSISTANT</span>
        </h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
          
          {/* AI Workout Generator */}
          <AiToolCard
            title="AI Workout Generator"
            description="Get a custom workout plan based on your goals and equipment."
            loading={workoutLoading}
            result={workoutResult}
            className="fade-in"
          >
            <form className="space-y-4" onSubmit={handleWorkoutSubmit}>
              <FormSelect name="goal" label="Your Goal">
                <option>Build Muscle</option>
                <option>Lose Fat</option>
                <option>Increase Stamina</option>
              </FormSelect>
              <FormSelect name="time" label="Time Available">
                <option>30 Minutes</option>
                <option>60 Minutes</option>
                <option>90+ Minutes</option>
              </FormSelect>
              <FormSelect name="equipment" label="Equipment">
                <option>Full Gym</option>
                <option>Dumbbells Only</option>
                <option>Bodyweight Only</option>
              </FormSelect>
              <button type="submit" className="btn-primary w-full flex items-center justify-center" disabled={workoutLoading}>
                {workoutLoading ? <><Loader className="mr-2 loader-dark" /> Generating...</> : 'Generate Workout'}
              </button>
            </form>
          </AiToolCard>
          
          {/* AI Nutrition Planner */}
          <AiToolCard
            title="AI Nutrition Planner"
            description="Get a sample meal plan tailored to your dietary needs."
            loading={nutritionLoading}
            result={nutritionResult}
            className="fade-in"
          >
            <form className="space-y-4" onSubmit={handleNutritionSubmit}>
              <FormSelect name="nutrition-goal" label="Your Goal">
                <option>Weight Loss</option>
                <option>Muscle Gain</option>
                <option>Maintenance</option>
              </FormSelect>
              <FormSelect name="preference" label="Dietary Preference">
                <option>Anything</option>
                <option>Vegetarian</option>
                <option>Vegan</option>
                <option>Low-Carb</option>
              </FormSelect>
              <button type="submit" className="btn-primary w-full flex items-center justify-center" disabled={nutritionLoading}>
                {nutritionLoading ? <><Loader className="mr-2 loader-dark" /> Generating...</> : 'Generate Meal Plan'}
              </button>
            </form>
          </AiToolCard>

        </div>
      </div>
    </section>
  );
};

const FormSelect = ({ name, label, children }) => (
  <div>
    <label htmlFor={name} className="block text-sm font-medium text-gray-300 mb-2">{label}</label>
    <select id={name} name={name} className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white focus:border-accent focus:ring-accent focus:ring-1 outline-none">
      {children}
    </select>
  </div>
);

const AiToolCard = ({ title, description, loading, result, children, className = '' }) => (
  <div className={`bg-gray-800 p-8 rounded-xl border border-gray-700 ${className}`}>
    <h3 className="text-3xl font-bold mb-3">{title}</h3>
    <p className="text-gray-400 mb-6">{description}</p>
    {children}
    {loading && (
      <div className="mt-6 text-center text-accent flex items-center justify-center gap-2">
        <Loader className="animate-spin" />
        <span>Generating your plan...</span>
      </div>
    )}
    {result && (
      <div className="mt-6 p-4 bg-gray-900 rounded-lg border border-gray-700 max-h-96 overflow-y-auto">
        <h4 className="font-bold text-accent mb-2">Your AI-Generated Plan:</h4>
        <p className="text-gray-300 whitespace-pre-wrap text-sm">{result}</p>
      </div>
    )}
  </div>
);

// --- Pricing Section ---
const PricingSection = () => (
  <section id="pricing" className="py-24 bg-gray-900">
    <div className="container mx-auto px-6">
      <h2 className="text-4xl md:text-5xl font-bold text-center mb-16 fade-in">
        CHOOSE YOUR <span className="text-accent">PLAN</span>
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-5xl mx-auto">
        <PricingCard 
          plan="Standard" 
          price="49"
          features={[
            { text: "Full Gym Access", included: true },
            { text: "Basic Locker", included: true },
            { text: "AI Workout Generator", included: true },
            { text: "Group Classes", included: false },
            { text: "Personal Training", included: false },
          ]}
        />
        <PricingCard 
          plan="Premium" 
          price="79"
          features={[
            { text: "Full Gym Access", included: true },
            { text: "Premium Locker", included: true },
            { text: "AI Workout Generator", included: true },
            { text: "All Group Classes", included: true },
            { text: "Personal Training", included: false },
          ]}
          featured={true}
        />
        <PricingCard 
          plan="Elite" 
          price="129"
          features={[
            { text: "Full Gym Access", included: true },
            { text: "Premium Locker", included: true },
            { text: "AI Workout Generator", included: true },
            { text: "All Group Classes", included: true },
            { text: "4x Personal Training/mo", included: true },
          ]}
        />
      </div>
    </div>
  </section>
);


// --- Contact Section ---
const ContactSection = () => {
  const [submitted, setSubmitted] = useState(false);

  const handleContactSubmit = (e) => {
    e.preventDefault();
    // Simulate form submission
    console.log("Contact form submitted");
    setSubmitted(true);
    setTimeout(() => setSubmitted(false), 3000); // Reset after 3 sec
    e.target.reset();
  };

  return (
    <section id="contact" className="py-24">
      <div className="container mx-auto px-6">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
          {/* Contact Form */}
          <div className="fade-in">
            <h2 className="text-4xl md:text-5xl font-bold text-left mb-4">
              GET IN <span className="text-accent">TOUCH</span>
            </h2>
            <p className="text-lg text-gray-400 mb-8">Have questions? Fill out the form and we'll get back to you shortly.</p>
            <form className="space-y-6" onSubmit={handleContactSubmit}>
              <FormInput name="name" label="Name" placeholder="Your Name" />
              <FormInput name="email" label="Email" type="email" placeholder="you@example.com" />
              <div>
                <label htmlFor="message" className="block text-sm font-medium text-gray-300 mb-2">Message</label>
                <textarea id="message" name="message" rows="5" className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white focus:border-accent focus:ring-accent focus:ring-1 outline-none" placeholder="Your message..."></textarea>
              </div>
              <button type="submit" className="btn-primary w-full">Send Message</button>
              {submitted && (
                <p className="text-center text-accent">Thanks for your message! We'll be in touch.</p>
              )}
            </form>
          </div>
          
          {/* --- THIS BLOCK IS UPDATED --- */}
          {/* Info Section (Map is removed) */}
          <div className="space-y-8 fade-in">
            <div className="text-lg bg-gray-800/50 border border-gray-700 p-8 rounded-lg space-y-4">
              <p className="flex items-start gap-3">
                <strong className="text-gray-100 w-20 flex-shrink-0">Address:</strong> 
                {/* Link to Google Maps with your coordinates */}
                <a 
                  href="https://www.google.com/maps?q=17.330304983478634,78.52553771138167" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-accent hover:underline"
                >
                  Jillelaguda New Venkateshwaracolony,Hyderabad
                  <br />
                  <span className="text-sm text-gray-400">(Click to view map)</span>
                </a>
              </p>
              <p className="flex items-center gap-3">
                <strong className="text-gray-100 w-20 flex-shrink-0">Phone:</strong> 
                {/* Clickable phone link */}
                <a href="tel:+11234567890" className="text-gray-300 hover:text-accent">+91 9182745442</a>
              </p>
              <p className="flex items-center gap-3">
                <strong className="text-gray-100 w-20 flex-shrink-0">Email:</strong> 
                {/* Clickable email link */}
                <a href="mailto:info@ascendgym.com" className="text-gray-300 hover:text-accent">rahulpv333175@gmail.com</a>
              </p>
            </div>
          </div>
          {/* --- END OF UPDATED BLOCK --- */}

        </div>
      </div>
    </section>
  );
};

const FormInput = ({ name, label, type = 'text', placeholder }) => (
  <div>
    <label htmlFor={name} className="block text-sm font-medium text-gray-300 mb-2">{label}</label>
    <input 
      type={type} 
      id={name} 
      name={name}
      className="w-full bg-gray-800 border border-gray-700 rounded-lg p-3 text-white focus:border-accent focus:ring-accent focus:ring-1 outline-none" 
      placeholder={placeholder}
      required 
    />
  </div>
);

export default HomePage;

