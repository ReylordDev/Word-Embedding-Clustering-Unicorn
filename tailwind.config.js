/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./templates/**/*.html",
            "./static/src/**/*.js"],
  theme: {
    extend: {
      colors: {
        'text': '#160622',
        'background': '#f9f4fd',
        'primary': '#932dd8',
        'secondary': '#e986c6',
        'accent': '#e15a84',
       },

    },
  },
  plugins: [],
}

