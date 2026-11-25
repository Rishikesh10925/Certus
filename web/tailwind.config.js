/** @type {import('tailwindcss').Config} */
const token = (name) => `rgb(var(--${name}) / <alpha-value>)`;

export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Certus Math"', '"Space Grotesk Variable"', '"Segoe UI"', "system-ui", "sans-serif"],
        mono: ['"Certus Math"', '"JetBrains Mono Variable"', "ui-monospace", '"Cascadia Mono"', "monospace"],
      },
      colors: {
        bg: token("bg"),
        surface: token("surface"),
        muted: token("muted"),
        sunken: token("sunken"),
        border: token("border"),
        "border-strong": token("border-strong"),
        fg: token("fg"),
        subtle: token("subtle"),
        faint: token("faint"),
        accent: token("accent"),
        "accent-fg": token("accent-fg"),
        "accent-ink": token("accent-ink"),
        "accent-soft": token("accent-soft"),
        clear: token("clear"),
        "clear-soft": token("clear-soft"),
        "clear-ink": token("clear-ink"),
        flag: token("flag"),
        "flag-soft": token("flag-soft"),
        "flag-ink": token("flag-ink"),
        escalate: token("escalate"),
        "escalate-soft": token("escalate-soft"),
        "escalate-ink": token("escalate-ink"),
        shift: token("shift"),
        "shift-ink": token("shift-ink"),
        defect: token("defect"),
        danger: token("danger"),
      },
      borderRadius: {
        xl: "14px",
      },
      boxShadow: {
        card: "var(--shadow-1)",
        pop: "var(--shadow-2)",
      },
      keyframes: {
        "fade-in": { from: { opacity: 0, transform: "translateY(4px)" }, to: { opacity: 1, transform: "none" } },
        rise: { from: { opacity: 0, transform: "translateY(10px)" }, to: { opacity: 1, transform: "none" } },
        shimmer: { "100%": { transform: "translateX(100%)" } },
        "slide-in-right": { from: { transform: "translateX(104%)" }, to: { transform: "none" } },
        "slide-in-left": { from: { transform: "translateX(-104%)" }, to: { transform: "none" } },
        "slide-in-up": { from: { transform: "translateY(104%)" }, to: { transform: "none" } },
        "pulse-ring": {
          "0%": { boxShadow: "0 0 0 0 rgb(var(--clear) / 0.6)" },
          "70%": { boxShadow: "0 0 0 7px rgb(var(--clear) / 0)" },
          "100%": { boxShadow: "0 0 0 0 rgb(var(--clear) / 0)" },
        },
      },
      animation: {
        "fade-in": "fade-in 240ms ease-out both",
        rise: "rise 600ms cubic-bezier(.2,.8,.2,1) both",
        shimmer: "shimmer 1.4s infinite",
        "slide-in-right": "slide-in-right 320ms cubic-bezier(.2,.8,.2,1) both",
        "slide-in-left": "slide-in-left 280ms cubic-bezier(.2,.8,.2,1) both",
        "slide-in-up": "slide-in-up 320ms cubic-bezier(.2,.8,.2,1) both",
        "pulse-ring": "pulse-ring 2.4s infinite",
      },
    },
  },
  plugins: [],
};
