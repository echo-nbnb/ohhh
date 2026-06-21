import React from "react";
import ReactDOM from "react-dom/client";
import Act1Entry from "./pages/Act1Entry/Act1Entry";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Act1Entry switchDelay={5000} />
  </React.StrictMode>,
);
