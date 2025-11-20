/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'accent': '#D4AF37', // Our gym's gold accent color
        'dark-bg': '#0B0B0F',
      },
      fontFamily: {
        inter: ['Inter', 'sans-serif'],
      },
      // For the modal and sidebar animations
      transitionProperty: {
        'transform': 'transform',
        'opacity': 'opacity',
      }
    },
  },
  plugins: [],
}