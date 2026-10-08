/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        canvas: "#F6F8FB",
        surface: "#FFFFFF",
        navy: {
          DEFAULT: "#153047",
          foreground: "#102235",
          muted: "#617286",
          light: "#EAEFF4",
        },
        border: "#E3E9EF",
        teal: {
          50: "#F0FDFA",
          100: "#CCFBF1",
          200: "#99F6E4",
          300: "#5EEAD4",
          400: "#2DD4BF",
          500: "#14B8A6",
          600: "#0D9488",
          700: "#0F766E",
          800: "#115E59",
          900: "#134E4A",
          950: "#042F2E",
        },
        clinical: {
          success: "#16794B",
          warning: "#B45309",
          danger: "#B42318",
        },
      },
    },
  },
  plugins: [],
}
