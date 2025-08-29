/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        gray_bg: "#1F1F1F",
        orange_accent: "#FF7F11",
      },
    },
  },
  plugins: [],
};
