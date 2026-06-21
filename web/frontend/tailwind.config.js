/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        avango: ['"Avango Display Serif"', "serif"],
        against: ['"Against"', "serif"],
        "serif-cn": ['"Source Han Serif CN"', '"Noto Serif SC"', "serif"],
      },
      colors: {
        ink: "#0d120e",
        panel: "#151d17",
        gold: "#c6a15b",
        paper: "#e8dec6",
      },
    },
  },
  plugins: [],
};
