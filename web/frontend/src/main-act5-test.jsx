import React, { useRef, useState, useCallback, useEffect } from "react";
import { createRoot } from "react-dom/client";
import Act5Postcard from "./pages/Act5Postcard/Act5Postcard";
import html2canvas from "html2canvas";

const MOCK = {
  colors: { primary: "#496b4a", secondary: "#8d3d36", primaryName: "岳麓绿", secondaryName: "书院红" },
  selectedPlaces: ["岳麓山", "湘江水"],
  title: { cn: "岳麓绿映书院红", en: "GREEN MEETS RED" },
  traceText: "[岳麓绿｜书院红] → [张栻]",
  imageryItems: [
    { id: "q1", name: "岳麓书院", imageUrl: "", description: ["你画下了岳麓书院。"], position: { left: 25, top: 58, width: 30 } },
    { id: "q2", name: "古树", imageUrl: "", description: ["你画下了古树。"], position: { left: 68, top: 42, width: 22 } },
  ],
  objectText: ["绿与红的相遇，书院与古树。"],
  aiWriting: ["岳麓绿已经展开。", "岳麓书院、古树也已经落下。", "张栻的声音回荡在千年书院中。", "这就是你寻到的千年色。"],
  person: { name: "张栻", portraitUrl: "" },
  mainTitleImageUrl: "",
  downloadQrUrl: "",
  createdAtText: "2026.06.22\n11:30",
};

function App() {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [log, setLog] = useState([]);
  const wrapperRef = useRef(null);
  const addLog = useCallback(msg => setLog(l => [...l, `${new Date().toLocaleTimeString("zh-CN",{hour12:false})} ${msg}`]), []);

  const doCapture = useCallback(async () => {
    const el = wrapperRef.current;
    if (!el) { addLog("ERROR: no ref"); return; }
    addLog("Starting capture of full Act5Postcard...");
    // Wait for CSS animations to complete
    addLog("Waiting 1s for animations...");
    await new Promise(r => setTimeout(r, 1000));
    // Force all elements to opacity:1 (animations may not have finished)
    el.querySelectorAll("*").forEach(e => { e.style.opacity = "1"; e.style.animation = "none"; });
    addLog(`Section size: ${el.offsetWidth}x${el.offsetHeight}, children: ${el.children.length}`);
    // Log first few children for context
    for (let i = 0; i < Math.min(5, el.children.length); i++) {
      const ch = el.children[i];
      addLog(`  child[${i}]: <${ch.tagName.toLowerCase()}> class="${ch.className.slice(0,40)}" size=${ch.offsetWidth}x${ch.offsetHeight}`);
    }
    try {
      // Rasterize SVGs — detailed logging
      const imgs = el.querySelectorAll("img");
      addLog(`=== Found ${imgs.length} img elements ===`);
      let svgCount = 0, failCount = 0, skipCount = 0;
      for (const img of imgs) {
        const src = img.src || img.getAttribute("src") || img.currentSrc || "";
        if (!src) { addLog(`  SKIP: empty src`); continue; }
        if (!src.includes(".svg")) { continue; }
        svgCount++;
        const w = img.offsetWidth || img.clientWidth || img.width || 300;
        const h = img.offsetHeight || img.clientHeight || img.height || 300;
        addLog(`  SVG#${svgCount}: ${w}x${h} natural=${img.naturalWidth}x${img.naturalHeight} src=${src.slice(-40)}`);
        if (w < 10 || h < 10) { skipCount++; addLog(`    SKIP: too small (${w}x${h})`); continue; }
        const result = await new Promise(resolve => {
          const c = document.createElement("canvas"); c.width = w; c.height = h;
          const ctx = c.getContext("2d"); const tmp = new Image();
          tmp.onload = () => { try { ctx.drawImage(tmp, 0, 0, w, h); const data = c.toDataURL(); img.src = data; resolve(`OK (${Math.round(data.length/1024)}KB)`); } catch(e) { resolve(`Draw err: ${e.message}`); } };
          tmp.onerror = (e) => { resolve(`Load err`); };
          tmp.src = src;
        });
        addLog(`    Result: ${result}`);
        if (result !== "OK") failCount++;
      }
      addLog(`=== Done: ${svgCount} SVGs, ${svgCount - failCount - skipCount} OK, ${failCount} failed, ${skipCount} skipped ===`);
      // Rasterize CSS SVG URLs (background-image, mask-image, etc.)
      const allEls = el.querySelectorAll("*");
      let cssSvgCount = 0;
      for (const elem of allEls) {
        const style = elem.style || {};
        const bg = style.background || style.backgroundImage || getComputedStyle(elem).backgroundImage;
        const mask = style.maskImage || style.WebkitMaskImage || getComputedStyle(elem).maskImage || getComputedStyle(elem).WebkitMaskImage;
        const urls = [];
        if (bg && bg.includes("url(") && bg.includes(".svg")) urls.push({ prop: "backgroundImage", value: bg });
        if (mask && mask.includes("url(") && mask.includes(".svg")) urls.push({ prop: "maskImage", value: mask });
        for (const { prop, value } of urls) {
          const match = value.match(/url\(["']?([^"')]*\.svg[^"')]*)["']?\)/);
          if (!match) continue;
          const svgUrl = match[1];
          cssSvgCount++;
          const r = elem.getBoundingClientRect();
          const w = r.width || elem.offsetWidth || 300;
          const h = r.height || elem.offsetHeight || 300;
          if (w < 5 || h < 5) continue;
          await new Promise(resolve => {
            const c = document.createElement("canvas"); c.width = w; c.height = h;
            const ctx = c.getContext("2d"); const tmp = new Image();
            tmp.onload = () => { try { ctx.drawImage(tmp, 0, 0, w, h); const data = c.toDataURL(); elem.style.setProperty(prop === "maskImage" ? "background" : "backgroundImage", `url(${data})`, "important"); elem.style.setProperty("maskImage", "none", "important"); elem.style.setProperty("-webkit-mask-image", "none", "important"); } catch(e){} resolve(); };
            tmp.onerror = () => resolve();
            tmp.src = svgUrl;
          });
        }
      }
      addLog(`Rasterized ${cssSvgCount} CSS SVG URLs`);
      // Try capturing WebGL directly (preserveDrawingBuffer:true should work now)
      const canvas = await html2canvas(el, { scale: 1, backgroundColor: "#f8f8f4", allowTaint: true, useCORS: true });
      addLog(`Captured: ${canvas.width}x${canvas.height}`);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.85);
      setPreviewUrl(dataUrl);
      addLog(`JPEG: ${Math.round(dataUrl.length/1024)}KB`);
    } catch(e) { addLog(`ERROR: ${e.message}`); }
  }, [addLog]);

  return React.createElement("div", { style: { width: "100vw", height: "100vh", overflow: "hidden" } },
    // Wrapper with ref around Act5
    React.createElement("div", { ref: wrapperRef, style: { width: "100%", height: "100%" } },
      React.createElement(Act5Postcard, {
        postcardData: MOCK,
        debugStage: 10,
        autoPlay: false,
      })
    ),
    // Controls
    React.createElement("div", { style: { position: "fixed", top: 10, right: 10, zIndex: 99999, display: "flex", gap: 8 } },
      React.createElement("button", {
        onClick: doCapture,
        style: { padding: "14px 24px", fontSize: 16, cursor: "pointer", background: "#4CAF50", color: "#fff", border: "none", borderRadius: 8, fontWeight: 700 }
      }, "📸 Capture Full Act5")
    ),
    // Preview
    previewUrl && React.createElement("img", {
      src: previewUrl,
      style: { position: "fixed", bottom: 10, right: 10, zIndex: 99998, maxWidth: 350, border: "2px solid #4CAF50", borderRadius: 8, boxShadow: "0 4px 20px rgba(0,0,0,0.5)" }
    }),
    // Log
    React.createElement("pre", { style: { position: "fixed", bottom: 10, left: 10, zIndex: 99999, width: 420, maxHeight: 220, overflow: "auto", background: "rgba(0,0,0,0.88)", color: "#0f0", fontSize: 11, padding: 10, borderRadius: 8 } },
      log.join("\n")
    )
  );
}

createRoot(document.getElementById("root")).render(React.createElement(App));
