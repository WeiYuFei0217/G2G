/* ==========================================================================
   viewer.js — 双模式交互式 3D 点云查看器(three.js,本地内嵌)

   Reloc 模式(5 例): 两序列重定位
     · View   : 拼接前 / 拼接后(Pred) / 拼接后(GT)  —— 控制 B 组点云的对齐
     · Color  : RGB 真彩 / A·B group 着色(cyan/orange)
     · Cameras: GT-vs-Pred 实心相机体对比(固定于 A0 系):
                A 锚点(橙)+ B-GT(绿)+ B-Pred(蓝);蓝绿越重合预测越准
   Rig 模式(N 例): 多相机 rig 里程计
     · Frames : 滑块选拼接前 K 帧(均按 pred 位姿累乘放置)
     · Shading: 当前帧提亮(其余灰度降亮) / 全帧 RGB
     · Cameras: 单目(每帧 1 相机体)/ 多目(每帧 K 相机体)
     · 相机   : 最新帧 pred(蓝)+ gt(绿)+ 上一帧(琥珀),均为实心相机体

   相机统一渲染为"实心彩色相机体"(半透明四棱锥 + 亮边 + 中心立方体),
   远比线框视锥醒目。配色与论文 case study(scene_gt_vs_pred.mlp)一致。

   坐标系: 数据为 OpenCV(Y-down),根节点绕 X 翻 180° 使显示 Y-up。
   由 main.js 在交互区进入视口时动态 import 并调用 initViewer()。
   ========================================================================== */
"use strict";

import * as THREE from "three";
import { OrbitControls } from "../vendor/three/OrbitControls.js";
import { PLYLoader } from "../vendor/three/PLYLoader.js";

const RELOC_ROOT = "./assets/pointclouds/reloc/";
const RIG_ROOT = "./assets/pointclouds/rig/";

const CYAN = 0x06b6d4, ORANGE = 0xf9731a;          // reloc 点云 A / B 着色
// 相机体 GT-vs-Pred 配色(与论文 case study scene_gt_vs_pred.mlp 一致, matplotlib tab10)
const CAM_A_GT = 0xff7f0e, CAM_A_PRED = 0xd62728;   // A 组锚点: GT 橙 / Pred 红
const CAM_B_GT = 0x2ca02c, CAM_B_PRED = 0x1f77b4;   // B 组: GT 绿 / Pred 蓝
const RIG_PRED = 0x3b82f6, RIG_GT = 0x10b981, RIG_PREV = 0xf59e0b;
const GRAY = 0x707888;

// 例子顺序与显示名(只显示 manifest 中存在的)
const RELOC_ORDER = ["hm3d", "tartanground", "nclt", "zjh", "zjh_real"];
const RELOC_LABEL = { hm3d: "HM3D", tartanground: "TartanGround", nclt: "NCLT",
                      zjh: "ZJH (sim)", zjh_real: "ZJH (real)" };
const RIG_ORDER = ["hm3d8", "hm3d4_15m", "tartanground", "nclt", "zjh_sim", "zjh_real"];
const RIG_LABEL = { hm3d8: "HM3D (8-cam)", hm3d4_15m: "HM3D (4-cam)",
                    tartanground: "TartanGround (4-cam)", nclt: "NCLT (5-cam)",
                    zjh_sim: "ZJH (sim, 4-cam)", zjh_real: "ZJH (real, 4-cam)" };

let renderer, scene, camera, controls, root;
let relocMan = {}, rigMan = {};
let raf = 0;

const state = {
  mode: "reloc",
  reloc: { ex: null, view: "pred", color: "rgb", cams: "both", ps: 2.0 },
  rig: { ex: null, view: "pred", shade: "highlight", cams: "mono", frames: 99, ps: 2.0 },
};
const relocCache = {};
const rigCache = {};
const loader = new PLYLoader();

/* ---------------------------------------------------- 矩阵工具 */
function mat4(rows) {
  const m = new THREE.Matrix4();
  m.set(rows[0][0], rows[0][1], rows[0][2], rows[0][3],
        rows[1][0], rows[1][1], rows[1][2], rows[1][3],
        rows[2][0], rows[2][1], rows[2][2], rows[2][3],
        rows[3][0], rows[3][1], rows[3][2], rows[3][3]);
  return m;
}

/* ============================================================ 入口 */
export async function initViewer() {
  const canvas = document.getElementById("viewer-canvas");
  if (!canvas) return;
  if (!hasWebGL()) return showFallback();

  [relocMan, rigMan] = await Promise.all([
    fetchJSON(RELOC_ROOT + "manifest.json"),
    fetchJSON(RIG_ROOT + "manifest.json"),
  ]);
  relocMan = relocMan || {};
  rigMan = rigMan || {};

  setupThree(canvas);
  bindModeTabs();
  bindControls();
  window.addEventListener("resize", onResize);
  await switchMode("reloc");
  loop();
}

function setupThree(canvas) {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  sizeRenderer();
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0b0d12);
  camera = new THREE.PerspectiveCamera(50, aspect(), 0.01, 8000);
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.rotateSpeed = 0.85;
  root = new THREE.Group();
  root.rotation.x = Math.PI;   // Y-down -> Y-up
  scene.add(root);
  scene.add(new THREE.AmbientLight(0xffffff, 1.0));
}

/* ============================================================ 模式/例子 tabs */
function bindModeTabs() {
  document.querySelectorAll("#viewer-mode-tabs .tab").forEach((t) =>
    t.addEventListener("click", () => {
      if (t.dataset.mode === state.mode) return;
      document.querySelectorAll("#viewer-mode-tabs .tab").forEach((x) =>
        x.setAttribute("aria-selected", String(x === t)));
      switchMode(t.dataset.mode);
    }));
}

function buildExampleTabs() {
  const host = document.getElementById("viewer-example-tabs");
  host.innerHTML = "";
  const order = state.mode === "reloc" ? RELOC_ORDER : RIG_ORDER;
  const label = state.mode === "reloc" ? RELOC_LABEL : RIG_LABEL;
  const man = state.mode === "reloc" ? relocMan : rigMan;
  const avail = order.filter((k) => man[k]);
  avail.forEach((k, i) => {
    const b = document.createElement("button");
    b.className = "tab";
    b.setAttribute("role", "tab");
    b.dataset.ex = k;
    b.textContent = label[k] || k;
    b.setAttribute("aria-selected", String(i === 0));
    b.addEventListener("click", () => selectExample(k));
    host.appendChild(b);
  });
  return avail[0] || null;
}

async function switchMode(mode) {
  state.mode = mode;
  // 控件面板显隐
  document.querySelectorAll(".mode-ctrl").forEach((el) =>
    el.hidden = el.dataset.mode !== mode);
  if (mode !== "rig") toggleMonoNote(false);   // 离开 rig 时收起单目提示
  // 隐藏所有已建对象
  Object.values(relocCache).forEach((e) => (e.group.visible = false));
  Object.values(rigCache).forEach((e) => (e.group.visible = false));
  const first = buildExampleTabs();
  if (first) await selectExample(first);
}

async function selectExample(ex) {
  document.querySelectorAll("#viewer-example-tabs .tab").forEach((t) =>
    t.setAttribute("aria-selected", String(t.dataset.ex === ex)));
  showLoading(true);
  try {
    if (state.mode === "reloc") { state.reloc.ex = ex; await showReloc(ex); }
    else { state.rig.ex = ex; await showRig(ex); }
  } catch (e) {
    console.error("example load failed", ex, e);
  }
  showLoading(false);
}

/* ============================================================ RELOC */
async function buildReloc(ds) {
  if (relocCache[ds]) return relocCache[ds];
  const meta = await fetchJSON(RELOC_ROOT + `${ds}.json`);
  const [geoA, geoB] = await Promise.all([
    loadPLY(RELOC_ROOT + meta.ply_a),
    loadPLY(RELOC_ROOT + meta.ply_b),
  ]);
  const ptsA = makePoints(geoA, CYAN);
  const ptsB = makePoints(geoB, ORANGE);

  // 尺度 & 视锥大小
  geoA.computeBoundingBox(); geoB.computeBoundingBox();
  const sizeA = new THREE.Vector3(), sizeB = new THREE.Vector3();
  geoA.boundingBox.getSize(sizeA); geoB.boundingBox.getSize(sizeB);
  const diag = Math.max(sizeA.length(), sizeB.length()) || 1;
  const baseSize = diag * 0.0016;
  // 相机体大小: 默认整体缩小; TartanGround 室外场景额外更小
  const CAM_K = { tartanground: 0.020 };
  const fscale = diag * (CAM_K[ds] || 0.033);
  ptsA.material.size = baseSize; ptsB.material.size = baseSize;

  // groupB: 仅含 B 点云,整体按所选 view 的 T_rel 变换(拼接前 / Pred / GT)
  const groupB = new THREE.Group();
  groupB.matrixAutoUpdate = false;
  groupB.add(ptsB);

  // 相机叠加层(固定于 A0 系): A 锚点 + B-GT(绿) + B-Pred(蓝) 的 GT-vs-Pred 对比。
  // 丰富 schema(每相机 pred/gt,如 case study)可直接用,否则由 T_rel 派生。
  const cam = buildRelocCams(meta, fscale);

  const g = new THREE.Group();
  g.add(ptsA, groupB, cam.layer);
  g.visible = false;
  root.add(g);

  relocCache[ds] = {
    group: g, ptsA, ptsB, groupB, cam, meta, baseSize,
    T_pred: mat4(meta.T_rel_pred), T_gt: mat4(meta.T_rel_gt),
  };
  return relocCache[ds];
}

async function showReloc(ds) {
  const e = await buildReloc(ds);
  Object.values(relocCache).forEach((x) => (x.group.visible = false));
  Object.values(rigCache).forEach((x) => (x.group.visible = false));
  e.group.visible = true;
  applyRelocView();
  applyRelocColor();
  applyRelocCams();
  applyPointSize();
  fitPointsRobust([e.ptsA, e.ptsB], 0.95);
  updateReadout();
  updateLegend();
}

function applyRelocView() {
  const e = relocCache[state.reloc.ex];
  if (!e) return;
  const v = state.reloc.view;
  const m = v === "before" ? new THREE.Matrix4() : (v === "gt" ? e.T_gt : e.T_pred);
  e.groupB.matrix.copy(m);
}

function applyRelocColor() {
  const e = relocCache[state.reloc.ex];
  if (!e) return;
  const grp = state.reloc.color === "group";
  setColorMode(e.ptsA, grp, CYAN);
  setColorMode(e.ptsB, grp, ORANGE);
}

// 相机叠加显隐: both|pred|gt|off。A 锚点在所有非 off 视图常驻;
// B-GT(绿)在 gt|both 显示, B-Pred(蓝)在 pred|both 显示;丰富 schema 下 A-Pred(红)同 pred。
function applyRelocCams() {
  const e = relocCache[state.reloc.ex];
  if (!e) return;
  const v = state.reloc.cams;
  const c = e.cam;
  c.layer.visible = v !== "off";
  c.gA.visible = v !== "off";
  c.gB_gt.visible = v === "gt" || v === "both";
  c.gB_pred.visible = v === "pred" || v === "both";
  if (c.gA_pred) c.gA_pred.visible = v === "pred" || v === "both";
}

/* ============================================================ RIG */
async function buildRig(name) {
  if (rigCache[name]) return rigCache[name];
  const meta = await fetchJSON(RIG_ROOT + `${name}.json`);
  const frames = meta.frames;
  // 双套点云: 融合多相机云(multi) + 仅前视单云(mono); 无 mono_ply 时回退融合云。
  const hasMono = frames.every((f) => f.mono_ply);
  let [geos, geosMono] = await Promise.all([
    Promise.all(frames.map((f) => loadPLY(RIG_ROOT + f.ply))),
    hasMono ? Promise.all(frames.map((f) => loadPLY(RIG_ROOT + f.mono_ply)))
            : Promise.resolve(null),
  ]);

  // 远点裁剪(如 NCLT: MapAnything 预测云含远处天空/楼宇离群点, 把场景对角撑到 200m+,
  // 而轨迹仅 ~10m, 帧显得很挤)。meta.clip_radius_m 存在时, 按【各帧本地相机系】保留
  // ||P|| <= r 的点(即只留观测相机 r 米内的近处结构), 与 before/pred/gt 三视图无关。
  if (meta.clip_radius_m) {
    const r = meta.clip_radius_m;
    geos = geos.map((g) => clipGeoRadius(g, r));
    if (geosMono) geosMono = geosMono.map((g) => clipGeoRadius(g, r));
  }

  // 尺度: 点大小按单帧尺寸;相机体按"单帧尺寸 vs 轨迹跨度"的较大者,
  // 这样拼接多帧后(视图拉远)相机体仍清晰可见,而非缩成一点。
  let diag = 1;
  geos[0].computeBoundingBox();
  const s0 = new THREE.Vector3(); geos[0].boundingBox.getSize(s0);
  diag = s0.length() || 1;
  const p0 = new THREE.Vector3(frames[0].pred[0][3], frames[0].pred[1][3], frames[0].pred[2][3]);
  let span = 0;
  frames.forEach((f) => {
    span = Math.max(span, p0.distanceTo(
      new THREE.Vector3(f.pred[0][3], f.pred[1][3], f.pred[2][3])));
  });
  const baseSize = diag * 0.0018;
  const fscale = Math.max(diag, span) * 0.04;

  const extr = meta.extrinsics.map(mat4);
  const g = new THREE.Group();
  const frameObjs = geos.map((geo, i) => {
    const pts = makePoints(geo, GRAY);
    pts.material.size = baseSize;
    const ptsMono = makePoints(geosMono ? geosMono[i] : geo, GRAY);
    ptsMono.material.size = baseSize;
    ptsMono.visible = false;
    const obj = new THREE.Group();
    obj.matrixAutoUpdate = false;
    obj.matrix.copy(mat4(frames[i].pred));   // 始终按 pred 放置
    obj.add(pts, ptsMono);
    g.add(obj);
    return { obj, pts, ptsMono, predM: mat4(frames[i].pred), gtM: mat4(frames[i].gt) };
  });
  const camLayer = new THREE.Group();
  g.add(camLayer);
  g.visible = false;
  root.add(g);

  rigCache[name] = { group: g, frameObjs, camLayer, extr, meta, baseSize, fscale,
                     n: frames.length };
  return rigCache[name];
}

async function showRig(name) {
  const e = await buildRig(name);
  Object.values(relocCache).forEach((x) => (x.group.visible = false));
  Object.values(rigCache).forEach((x) => (x.group.visible = false));
  e.group.visible = true;
  // 滑块上限
  const slider = document.getElementById("rig-frames");
  slider.max = String(e.n);
  if (state.rig.frames > e.n) state.rig.frames = e.n;
  slider.value = String(state.rig.frames);
  document.getElementById("rig-frames-val").textContent = String(state.rig.frames);
  applyRigAll();
  fitRig(e);
  updateReadout();
  updateLegend();
}

function curK() {
  const e = rigCache[state.rig.ex];
  return Math.max(1, Math.min(state.rig.frames, e ? e.n : 1));
}

// mono 模式: 只渲前视单云(避免多相机点云全叠加杂乱); multi 模式: 渲融合多相机云。
function rigMono() { return state.rig.cams === "mono"; }
function activePts(f) { return rigMono() ? f.ptsMono : f.pts; }

function applyRigAll() {
  applyRigFrames();
  applyRigShade();
  applyRigCams();
  applyPointSize();
}

function applyRigFrames() {
  const e = rigCache[state.rig.ex];
  if (!e) return;
  const K = curK();
  const mono = rigMono();
  const view = state.rig.view;
  e.frameObjs.forEach((f, i) => {
    f.obj.visible = i < K;
    f.pts.visible = !mono;
    f.ptsMono.visible = mono;
    f.obj.matrix.copy(rigWorld(f, view));   // 按当前 merge 视图摆放(before/pred/gt)
  });
}

function applyRigShade() {
  const e = rigCache[state.rig.ex];
  if (!e) return;
  const K = curK();
  const hi = state.rig.shade === "highlight";
  e.frameObjs.forEach((f, i) => {
    if (i >= K) return;
    const pts = activePts(f);             // 仅给当前激活云(mono/multi)上色
    const current = i === K - 1;
    if (!hi || current) {
      setColorMode(pts, false, 0xffffff); // RGB
      pts.material.opacity = 1.0; pts.material.transparent = false;
    } else {
      setColorMode(pts, true, GRAY);      // 灰度
      pts.material.opacity = 0.4; pts.material.transparent = true;
    }
    pts.material.needsUpdate = true;
  });
}

function applyRigCams() {
  const e = rigCache[state.rig.ex];
  if (!e) return;
  const K = curK();
  e.camLayer.clear();
  const mono = rigMono();
  const cams = mono ? [e.meta.mono_cam || 0] : e.extr.map((_, j) => j);
  toggleMonoNote(mono);                   // 右上角"仅渲染单目"提示
  const fs = e.fscale;
  const view = state.rig.view;
  // 【单步相对位姿误差】可视化(而非累计漂移):
  //   把当前帧的 pred 与 gt 都重锚到【上一帧】的世界位姿(anchor), 再各自乘上
  //   prev->last 的单步相对变换。这样 琥珀(anchor)→蓝(本步pred) 与 →绿(本步gt)
  //   共用同一锚点, 绿↔蓝只剩单步预测误差。anchor 跟随 merge 视图:
  //   pred 视图 anchor=predM[prev](蓝即 predM[last], 绿=重锚真值); gt 视图反之; before=原点。
  const last = K - 1;
  const prev = Math.max(0, last - 1);
  const fl = e.frameObjs[last], fp = e.frameObjs[prev];
  const anchor = rigWorld(fp, view);
  const stepPred = new THREE.Matrix4().multiplyMatrices(
    new THREE.Matrix4().copy(fp.predM).invert(), fl.predM);
  const stepGT = new THREE.Matrix4().multiplyMatrices(
    new THREE.Matrix4().copy(fp.gtM).invert(), fl.gtM);
  const predWorld = new THREE.Matrix4().multiplyMatrices(anchor, stepPred);
  const gtWorld = new THREE.Matrix4().multiplyMatrices(anchor, stepGT);
  // gt 体略大成同心外壳: 预测准时绿包蓝清晰可辨, 发散时两体自然分离(避免 z-fighting)。
  cams.forEach((j) => {
    e.camLayer.add(cameraBody(new THREE.Matrix4().multiplyMatrices(gtWorld, e.extr[j]), RIG_GT, fs * 1.18));
    e.camLayer.add(cameraBody(new THREE.Matrix4().multiplyMatrices(predWorld, e.extr[j]), RIG_PRED, fs));
    if (K >= 2)   // 上一帧锚点(琥珀), 略小以区分历史
      e.camLayer.add(cameraBody(new THREE.Matrix4().multiplyMatrices(anchor, e.extr[j]), RIG_PREV, fs * 0.85));
  });
}

function fitRig(e) {
  const K = curK();
  const pts = [];
  for (let i = 0; i < K; i++) pts.push(activePts(e.frameObjs[i]));
  fitPointsRobust(pts, 0.9);
}

// 右上角提示: mono 模式下说明"仅渲染前视单目点云, 位姿推理仍用完整 rig"。
function toggleMonoNote(show) {
  const el = document.getElementById("viewer-mono-note");
  if (el) el.hidden = !(state.mode === "rig" && show);
}

/* ============================================================ 几何构建 */
// 按本地原点(相机中心)半径裁剪点云: 仅保留 ||P|| <= r 的点, 同步保留 color 属性。
// 用于去除全景重建里把包围盒撑大的稀疏远点(天空/远墙)。返回新 BufferGeometry。
function clipGeoRadius(geo, r) {
  const pos = geo.getAttribute("position");
  if (!pos) return geo;
  const col = geo.getAttribute("color");
  const r2 = r * r;
  const P = [], C = [];
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i), y = pos.getY(i), z = pos.getZ(i);
    if (x * x + y * y + z * z <= r2) {
      P.push(x, y, z);
      if (col) C.push(col.getX(i), col.getY(i), col.getZ(i));
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(P), 3));
  if (col) g.setAttribute("color", new THREE.BufferAttribute(new Float32Array(C), 3));
  return g;
}

const RIG_IDENTITY = new THREE.Matrix4();
// 某帧在当前 merge 视图下的世界摆放矩阵:
//   before = 单位阵(各帧云堆叠在原点, 未配准的"原始"态)
//   pred   = predM[i](里程计按【预测】相对位姿拼接, 即方法输出的地图)
//   gt     = gtM[i] (按【真值】相对位姿拼接, 完美参考地图)
function rigWorld(f, view) {
  if (view === "gt") return f.gtM;
  if (view === "before") return RIG_IDENTITY;
  return f.predM;
}

function makePoints(geo, fallback) {
  const hasColor = !!geo.getAttribute("color");
  const mat = new THREE.PointsMaterial({
    size: 0.01, sizeAttenuation: true,
    vertexColors: hasColor, color: hasColor ? 0xffffff : fallback,
  });
  const p = new THREE.Points(geo, mat);
  p.userData.hasColor = hasColor;
  return p;
}

function setColorMode(pts, uniform, color) {
  if (uniform) {
    pts.material.vertexColors = false;
    pts.material.color.setHex(color);
  } else {
    pts.material.vertexColors = pts.userData.hasColor;
    pts.material.color.setHex(pts.userData.hasColor ? 0xffffff : color);
  }
  pts.material.needsUpdate = true;
}

// 构建 reloc 相机叠加层(固定于 A0 系)。
//   A 锚点(橙) + B-GT(绿) + B-Pred(蓝);丰富 schema 额外支持 A-Pred(红)。
//   简化 schema: B-GT = T_rel_gt @ posesB, B-Pred = T_rel_pred @ posesB;
//   丰富 schema: 直接用 meta.posesB_in_a0 / pred_posesB_in_a0 / pred_posesA。
// 返回 { layer, gA, gA_pred, gB_gt, gB_pred, hasPred }。
function buildRelocCams(meta, fscale) {
  const Tg = mat4(meta.T_rel_gt), Tp = mat4(meta.T_rel_pred);
  const A_gt = meta.posesA.map(mat4);
  const A_pred = meta.pred_posesA ? meta.pred_posesA.map(mat4) : null;
  const B_gt = meta.posesB_in_a0
    ? meta.posesB_in_a0.map(mat4)
    : meta.posesB.map((m) => new THREE.Matrix4().multiplyMatrices(Tg, mat4(m)));
  const B_pred = meta.pred_posesB_in_a0
    ? meta.pred_posesB_in_a0.map(mat4)
    : meta.posesB.map((m) => new THREE.Matrix4().multiplyMatrices(Tp, mat4(m)));

  const gA = new THREE.Group();
  A_gt.forEach((m) => gA.add(cameraBody(m, CAM_A_GT, fscale)));
  const gA_pred = A_pred ? new THREE.Group() : null;
  if (gA_pred) A_pred.forEach((m) => gA_pred.add(cameraBody(m, CAM_A_PRED, fscale)));
  const gB_gt = new THREE.Group();
  B_gt.forEach((m) => gB_gt.add(cameraBody(m, CAM_B_GT, fscale)));
  const gB_pred = new THREE.Group();
  B_pred.forEach((m) => gB_pred.add(cameraBody(m, CAM_B_PRED, fscale)));

  const layer = new THREE.Group();
  layer.add(gA, gB_gt, gB_pred);
  if (gA_pred) layer.add(gA_pred);
  return { layer, gA, gA_pred, gB_gt, gB_pred, hasPred: !!A_pred };
}

// 由 Matrix4(cam->world)构建一个"实心彩色相机体":
//   半透明填充的四棱锥(顶点在相机中心,底面为像平面)+ 不透明亮边线框 + 中心小立方体。
// 比线框视锥醒目得多,用于在点云上清晰标出相机位姿。返回一个已摆放的 Group。
function cameraBody(m4, color, s) {
  const grp = new THREE.Group();
  const w = s * 0.8, h = s * 0.6, d = s;
  const a = [0, 0, 0], c0 = [-w, -h, d], c1 = [w, -h, d], c2 = [w, h, d], c3 = [-w, h, d];

  // (1) 半透明填充: 四个侧面 + 像平面(双面,关闭深度写入以免遮挡点云)
  const tri = [a, c0, c1, a, c1, c2, a, c2, c3, a, c3, c0, c0, c1, c2, c0, c2, c3];
  const fgeo = new THREE.BufferGeometry();
  fgeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(tri.flat()), 3));
  grp.add(new THREE.Mesh(fgeo, new THREE.MeshBasicMaterial({
    color, transparent: true, opacity: 0.26, side: THREE.DoubleSide, depthWrite: false,
  })));

  // (2) 不透明亮边
  const seg = [a, c0, a, c1, a, c2, a, c3, c0, c1, c1, c2, c2, c3, c3, c0];
  const lgeo = new THREE.BufferGeometry();
  lgeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(seg.flat()), 3));
  grp.add(new THREE.LineSegments(lgeo, new THREE.LineBasicMaterial({ color })));

  // (3) 相机中心实心小立方体(强调相机位置)
  grp.add(new THREE.Mesh(
    new THREE.BoxGeometry(s * 0.2, s * 0.2, s * 0.2),
    new THREE.MeshBasicMaterial({ color })));

  grp.applyMatrix4(m4);
  return grp;
}

/* ============================================================ 取景 */
// 稳健取景: 采样可见点云,逐轴中位数为中心,距中心距离的 pct 分位数为半径。
// 全景重建常含少量远墙点会把包围盒撑大,用分位数可聚焦主体而不被稀疏远点带偏。
function fitPointsRobust(ptsArr, pct) {
  const xs = [], ys = [], zs = [];
  const v = new THREE.Vector3();
  ptsArr.forEach((p) => {
    if (!p) return;
    p.updateWorldMatrix(true, false);
    const pos = p.geometry.getAttribute("position");
    if (!pos) return;
    const stride = Math.max(1, Math.floor(pos.count / 2500));
    for (let i = 0; i < pos.count; i += stride) {
      v.fromBufferAttribute(pos, i).applyMatrix4(p.matrixWorld);
      xs.push(v.x); ys.push(v.y); zs.push(v.z);
    }
  });
  if (!xs.length) return;
  const med = (a) => { const b = a.slice().sort((p, q) => p - q); return b[b.length >> 1]; };
  const center = new THREE.Vector3(med(xs), med(ys), med(zs));
  const d = xs.map((_, i) => {
    const dx = xs[i] - center.x, dy = ys[i] - center.y, dz = zs[i] - center.z;
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }).sort((p, q) => p - q);
  const radius = d[Math.min(d.length - 1, Math.floor(d.length * pct))] || 1;
  applyCamera(center, radius);
}

function applyCamera(center, radius) {
  const dist = radius / Math.sin((camera.fov * Math.PI) / 360);
  camera.near = Math.max(radius / 1000, 0.001);
  camera.far = dist * 20;
  camera.updateProjectionMatrix();
  const dir = new THREE.Vector3(0.7, 0.5, 1).normalize();
  camera.position.copy(center).addScaledVector(dir, dist * 1.05);
  controls.target.copy(center);
  controls.minDistance = radius * 0.1;
  controls.maxDistance = dist * 8;
  controls.update();
}

/* ============================================================ 控件绑定 */
function bindControls() {
  segHandler("#reloc-view", "v", (v) => { state.reloc.view = v; applyRelocView(); updateReadout(); });
  segHandler("#reloc-color", "v", (v) => { state.reloc.color = v; applyRelocColor(); });
  segHandler("#reloc-cams", "v", (v) => { state.reloc.cams = v; applyRelocCams(); });
  // rig merge 视图: Pred 拼接图 ↔ GT 拼接图 — 仅重摆点云 + 重绘相机, 【不重取景】,
  // 让相机固定, 用户原地切换即可直接对比两张拼接图的几何差异(与 reloc 切换一致)。
  segHandler("#rig-view", "v", (v) => {
    state.rig.view = v;
    applyRigFrames(); applyRigShade(); applyRigCams();
    updateLegend();
  });
  segHandler("#rig-shade", "v", (v) => { state.rig.shade = v; applyRigShade(); });
  // mono/multi 切换: 换激活点云(融合↔前视单云) + 重新上色/取景 + 切相机体与提示。
  segHandler("#rig-cams", "v", (v) => {
    state.rig.cams = v;
    applyRigFrames(); applyRigShade(); applyRigCams(); applyPointSize();
    const e = rigCache[state.rig.ex];
    if (e) fitRig(e);
  });
  segHandler("#point-size", "ps", (v) => {
    const f = parseFloat(v);
    if (state.mode === "reloc") state.reloc.ps = f; else state.rig.ps = f;
    applyPointSize();
  });
  const slider = document.getElementById("rig-frames");
  if (slider) slider.addEventListener("input", () => {
    state.rig.frames = parseInt(slider.value, 10);
    document.getElementById("rig-frames-val").textContent = slider.value;
    applyRigFrames(); applyRigShade(); applyRigCams(); updateReadout();
  });
}

function segHandler(sel, key, fn) {
  const seg = document.querySelector(sel);
  if (!seg) return;
  seg.querySelectorAll("button").forEach((b) =>
    b.addEventListener("click", () => {
      seg.querySelectorAll("button").forEach((x) =>
        x.setAttribute("aria-pressed", String(x === b)));
      fn(b.dataset[key]);
    }));
}

function applyPointSize() {
  if (state.mode === "reloc") {
    const e = relocCache[state.reloc.ex];
    if (!e) return;
    const sz = e.baseSize * state.reloc.ps;
    e.ptsA.material.size = sz; e.ptsB.material.size = sz;
  } else {
    const e = rigCache[state.rig.ex];
    if (!e) return;
    const sz = e.baseSize * state.rig.ps;
    e.frameObjs.forEach((f) => { f.pts.material.size = sz; f.ptsMono.material.size = sz; });
  }
}

/* ============================================================ 读数/图例 */
function updateReadout() {
  const box = document.getElementById("viewer-readout");
  const title = document.getElementById("ro-title");
  const err = document.getElementById("ro-err");
  if (!box) return;
  box.hidden = false;
  // 标题行按用户要求去掉(两模式皆然); 误差排成单行。
  if (title) title.textContent = "";
  if (state.mode === "reloc") {
    const e = relocCache[state.reloc.ex];
    if (!e) return;
    const er = e.meta.errors;
    err.innerHTML = `rot&nbsp;<b>${er.rot_deg.toFixed(2)}°</b> · `
      + `trans&nbsp;<b>${(er.trans_m * 100).toFixed(1)} cm</b> · overlap&nbsp;${(er.overlap ?? 0).toFixed(2)}`;
  } else {
    const e = rigCache[state.rig.ex];
    if (!e) return;
    const m = e.meta.metrics;
    // drift 指标偏大易误读, 按用户要求不显示; 保留 ATE 与 RPE(旋转°/平移cm 两轴误差)。
    err.innerHTML = `ATE&nbsp;<b>${(m.ate_rmse_m * 100).toFixed(1)} cm</b> · `
      + `RPE&nbsp;${m.per_step_rot_med.toFixed(2)}°/${(m.per_step_trans_med * 100).toFixed(1)}cm`;
  }
}

function updateLegend() {
  const el = document.getElementById("viewer-legend");
  if (!el) return;
  if (state.mode === "reloc") {
    const e = relocCache[state.reloc.ex];
    const rich = !!(e && e.cam && e.cam.hasPred);
    el.innerHTML =
      `<span><i style="background:#ff7f0e"></i>Group A${rich ? " · GT" : " (anchor)"}</span>`
      + (rich ? `<span><i style="background:#d62728"></i>Group A · Pred</span>` : "")
      + `<span><i style="background:#2ca02c"></i>Group B · GT</span>`
      + `<span><i style="background:#1f77b4"></i>Group B · Pred</span>`
      + `<span>Points: RGB or A/B color</span>`;
  } else {
    // 相机三元组体现【单步】相对位姿误差: 蓝/绿都相对琥珀(上一帧)锚定, 绿↔蓝即本步预测误差。
    el.innerHTML =
      `<span><i style="background:#f59e0b"></i>Previous frame (anchor)</span>`
      + `<span><i style="background:#3b82f6"></i>Latest · predicted step</span>`
      + `<span><i style="background:#10b981"></i>Latest · GT step</span>`;
  }
}

/* ============================================================ 渲染/工具 */
function loop() { raf = requestAnimationFrame(loop); controls.update(); renderer.render(scene, camera); }
function aspect() { const c = renderer.domElement; return c.clientWidth / Math.max(c.clientHeight, 1); }
function sizeRenderer() { const c = renderer.domElement; renderer.setSize(c.clientWidth, c.clientHeight, false); }
function onResize() { if (!renderer) return; sizeRenderer(); camera.aspect = aspect(); camera.updateProjectionMatrix(); }
function loadPLY(url) { return new Promise((res, rej) => loader.load(url, res, undefined, rej)); }
async function fetchJSON(url) {
  try { const r = await fetch(url, { cache: "no-cache" }); return r.ok ? await r.json() : null; }
  catch { return null; }
}
function showLoading(on) { const el = document.getElementById("viewer-loading"); if (el) el.style.display = on ? "flex" : "none"; }
function showFallback() {
  const fb = document.getElementById("viewer-fallback"), ld = document.getElementById("viewer-loading");
  if (ld) ld.style.display = "none";
  if (fb) fb.style.display = "flex";
}
function hasWebGL() {
  try {
    const c = document.createElement("canvas");
    return !!(window.WebGLRenderingContext && (c.getContext("webgl") || c.getContext("experimental-webgl")));
  } catch { return false; }
}
