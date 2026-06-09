/* ==========================================================================
   main.js — 页面交互(纯原生 ES module,无框架)
     · 移动端导航开合
     · 数据集 pill 标签页:切换视频源/封面 + 定性图
     · 异步注入响应式表格(data/tables/*.html)
     · BibTeX 一键复制
     · 进入交互区时再懒加载 three.js 查看器(viewer.js)
   ========================================================================== */
"use strict";

/* ----------------------------------------------------- 移动端导航开合 */
const navToggle = document.getElementById("navToggle");
const navLinks = document.getElementById("navlinks");
if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => navLinks.classList.toggle("open"));
  navLinks.querySelectorAll("a").forEach((a) =>
    a.addEventListener("click", () => navLinks.classList.remove("open"))
  );
}

/* ----------------------------------------------- 数据集标签页(视频+图) */
function setupTabGroup(group) {
  const vidSel = group.dataset.vid;          // 例 "#reloc-video"
  const vidBase = group.dataset.vidbase;     // 例 "reloc" -> assets/videos/reloc-<ds>.mp4
  const qualSel = group.dataset.qual;        // 可选,定性图 <img>
  const qualBase = group.dataset.qualbase;   // 例 "qual_reloc" -> assets/figures/qual_reloc_<ds>.png
  const video = vidSel ? document.querySelector(vidSel) : null;
  const qual = qualSel ? document.querySelector(qualSel) : null;
  const tabs = [...group.querySelectorAll(".tab")];

  function select(ds) {
    tabs.forEach((t) => t.setAttribute("aria-selected", String(t.dataset.ds === ds)));
    if (video) {
      const src = `./assets/videos/${vidBase}-${ds}.mp4`;
      const poster = `./assets/videos/${vidBase}-${ds}.jpg`;
      const source = video.querySelector("source");
      if (source.getAttribute("src") !== src) {
        source.setAttribute("src", src);
        video.setAttribute("poster", poster);
        video.load();
        // autoplay 属性会在 load 后自动尝试;部分浏览器需显式 play()
        video.play().catch(() => {});
      }
    }
    if (qual && qualBase) {
      qual.setAttribute("src", `./assets/figures/${qualBase}_${ds}.png`);
    }
  }

  tabs.forEach((t) =>
    t.addEventListener("click", () => select(t.dataset.ds))
  );
}
document.querySelectorAll("[data-tabgroup]").forEach(setupTabGroup);

/* --------------------------------------------- 异步注入响应式表格片段 */
async function injectTable(container) {
  const name = container.dataset.table; // reloc / rig / ablation
  try {
    const res = await fetch(`./data/tables/${name}.html`, { cache: "no-cache" });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    container.innerHTML = await res.text();
  } catch (err) {
    container.innerHTML =
      `<p class="cap">Table failed to load (${err.message}). ` +
      `See the table in the <a href="https://arxiv.org/abs/2606.08284" target="_blank" rel="noopener">arXiv paper</a>.</p>`;
  }
}
document.querySelectorAll("[data-table]").forEach(injectTable);

/* ----------------------------------------------------- BibTeX 一键复制 */
const copyBtn = document.getElementById("copyBib");
if (copyBtn) {
  copyBtn.addEventListener("click", async () => {
    const text = document.getElementById("bibtex-text").innerText;
    const label = copyBtn.querySelector("span");
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // 退化方案:选区复制
      const r = document.createRange();
      r.selectNodeContents(document.getElementById("bibtex-text"));
      const sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(r);
      document.execCommand("copy");
      sel.removeAllRanges();
    }
    copyBtn.classList.add("done");
    label.textContent = "Copied!";
    setTimeout(() => {
      copyBtn.classList.remove("done");
      label.textContent = "Copy";
    }, 1800);
  });
}

/* ------------------------------------ 导航高亮当前区块(scroll spy) */
const spyLinks = [...document.querySelectorAll(".nav .links a")];
const spyMap = new Map();
spyLinks.forEach((a) => {
  const id = a.getAttribute("href").slice(1);
  const sec = document.getElementById(id);
  if (sec) spyMap.set(sec, a);
});
if (spyMap.size) {
  const spy = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          spyLinks.forEach((l) => l.classList.remove("active"));
          spyMap.get(e.target)?.classList.add("active");
        }
      });
    },
    { rootMargin: "-45% 0px -50% 0px" }
  );
  spyMap.forEach((_, sec) => spy.observe(sec));
}

/* ------------------------- 进入交互区时再懒加载 three.js 查看器 ------- */
const interactive = document.getElementById("interactive");
if (interactive) {
  const once = new IntersectionObserver(
    (entries, obs) => {
      if (entries.some((e) => e.isIntersecting)) {
        obs.disconnect();
        import("./viewer.js")
          .then((m) => m.initViewer())
          .catch((err) => {
            console.error("viewer load failed", err);
            const fb = document.getElementById("viewer-fallback");
            const ld = document.getElementById("viewer-loading");
            if (ld) ld.style.display = "none";
            if (fb) fb.style.display = "flex";
          });
      }
    },
    { rootMargin: "200px" }
  );
  once.observe(interactive);
}
