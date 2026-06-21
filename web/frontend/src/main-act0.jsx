import React from "react";
import ReactDOM from "react-dom/client";
import Act0 from "./pages/Act0/Act0";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Act0 onNext={() => console.log("进入第一幕")} />
  </React.StrictMode>,
);
