import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";
import App from "./App";
import { ConfigProvider } from "./context/ConfigContext";
import { ErrorBoundary } from "./components/ErrorBoundary";

// Guard against a missing root element rather than silently crashing.
// WHY throw instead of console.error: a missing #root means index.html is
// broken. Throwing surfaces the problem immediately rather than producing
// a confusing "cannot read properties of null" stack trace later.
const root = document.getElementById("root");
if (!root) throw new Error("Root element #root not found in index.html");

createRoot(root).render(
  <StrictMode>
    <ErrorBoundary>
      <ConfigProvider>
        <App />
      </ConfigProvider>
    </ErrorBoundary>
  </StrictMode>,
);
