/** @type {import('tailwindcss').Config} */
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Certus Math"', '"Inter Variable"', "system-ui", "sans-serif"],
        mono: ['"Certus Math"', '"JetBrains Mono Variable"', "ui-monospace", "monospace"],
      },
      colors: {
        bg: token("bg"),
        surface: token("surface"),
        muted: token("muted"),
        border: token("border"),
        fg: token("fg"),
        subtle: token("subtle"),
        accent: token("accent"),
        "accent-fg": token("accent-fg"),
        clear: token("clear"),
        flag: token("flag"),
        escalate: token("escalate"),
        shift: token("shift"),
        defect: token("defect"),
      },
      boxShadow: {
        card: "0 1px 2px rgb(0 0 0 / 0.04), 0 1px 1px rgb(0 0 0 / 0.02)",
        pop: "0 12px 32px -8px rgb(0 0 0 / 0.25)",
      },
      keyframes: {
        "fade-in": { from: { opacity: 0, transform: "translateY(4px)" }, to: { opacity: 1, transform: "none" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: {
        "fade-in": "fade-in 240ms ease-out both",
        shimmer: "shimmer 1.4s infinite",
      },
    },
  },
  plugins: [],
};
