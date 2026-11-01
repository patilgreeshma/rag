/* ─────────────────────────────────────────────────────────────────────────
   FinRule AI – Frontend JavaScript
   Connects to the FastAPI backend at http://localhost:8000
───────────────────────────────────────────────────────────────────────── */

const API_BASE = "http://localhost:8000";

// ── DOM refs ───────────────────────────────────────────────────────────────
const messagesContainer = document.getElementById("messagesContainer");
const welcomeScreen     = document.getElementById("welcomeScreen");
const questionInput     = document.getElementById("questionInput");
const sendBtn           = document.getElementById("sendBtn");
const charCount         = document.getElementById("charCount");
const statusBadge       = document.getElementById("statusBadge");
const vectorCount       = document.getElementById("vectorCount");
const clearBtn          = document.getElementById("clearBtn");
const newChatBtn        = document.getElementById("newChatBtn");
const sidebar           = document.getElementById("sidebar");
const sidebarToggle     = document.getElementById("sidebarToggle");

// ── State ──────────────────────────────────────────────────────────────────
let isLoading = false;

// ── Init ───────────────────────────────────────────────────────────────────
(async function init() {
  await checkHealth();
  setupEventListeners();
  autoResizeTextarea();
})();

// ── Health check ───────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res  = await fetch(`${API_BASE}/health`);
    const data = await res.json();
    statusBadge.textContent = "● Online";
    statusBadge.className   = "header-badge online";
    vectorCount.querySelector("span:last-child").textContent =
      `${data.index_vectors.toLocaleString()} vectors indexed`;
  } catch {
    statusBadge.textContent = "● Offline – start backend";
    statusBadge.className   = "header-badge error";
    vectorCount.querySelector("span:last-child").textContent = "Backend unreachable";
  }
}

// ── Event listeners ────────────────────────────────────────────────────────
function setupEventListeners() {
  // Send on Enter (Shift+Enter = newline)
  questionInput.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled) handleSend();
    }
  });

  questionInput.addEventListener("input", () => {
    const len = questionInput.value.length;
    charCount.textContent = `${len} / 2000`;
    charCount.className   = len > 1800 ? "char-count warning" : "char-count";
    sendBtn.disabled      = len < 3 || isLoading;
    autoResizeTextarea();
  });

  sendBtn.addEventListener("click", handleSend);

  clearBtn.addEventListener("click", () => {
    messagesContainer.innerHTML = "";
    messagesContainer.appendChild(welcomeScreen);
    welcomeScreen.style.display = "";
  });

  newChatBtn.addEventListener("click", () => {
    messagesContainer.innerHTML = "";
    messagesContainer.appendChild(welcomeScreen);
    welcomeScreen.style.display = "";
    questionInput.value = "";
    charCount.textContent = "0 / 2000";
    sendBtn.disabled = true;
  });

  sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("collapsed"));

  // Example / chip buttons (delegated)
  document.addEventListener("click", e => {
    const btn = e.target.closest("[data-q]");
    if (btn) {
      const q = btn.getAttribute("data-q");
      questionInput.value = q;
      questionInput.dispatchEvent(new Event("input"));
      questionInput.focus();
      if (window.innerWidth <= 680) sidebar.classList.add("collapsed");
    }
  });
}

// ── Auto-resize textarea ──────────────────────────────────────────────────
function autoResizeTextarea() {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 160) + "px";
}

// ── Main send handler ─────────────────────────────────────────────────────
async function handleSend() {
  const question = questionInput.value.trim();
  if (!question || isLoading) return;

  // Hide welcome screen on first message
  if (welcomeScreen.parentNode) {
    welcomeScreen.style.display = "none";
  }

  // Append user message
  appendMessage("user", question);

  // Reset input
  questionInput.value = "";
  questionInput.style.height = "auto";
  charCount.textContent = "0 / 2000";
  sendBtn.disabled = true;

  // Show typing indicator
  const typingEl = appendTyping();

  isLoading = true;

  try {
    const res = await fetch(`${API_BASE}/query`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ question, top_k: 5 }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();
    typingEl.remove();
    appendAIMessage(data);

  } catch (err) {
    typingEl.remove();
    appendMessage("ai", `⚠️ Error: ${err.message}. Make sure the backend is running.`);
    showToast(err.message);
  } finally {
    isLoading = false;
    sendBtn.disabled = questionInput.value.length < 3;
    scrollToBottom();
  }
}

// ── DOM builders ──────────────────────────────────────────────────────────

function appendMessage(role, text) {
  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "👤" : "⚖️";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const answerEl = document.createElement("div");
  answerEl.className = "answer-text";
  answerEl.textContent = text;
  bubble.appendChild(answerEl);

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  messagesContainer.appendChild(msg);
  scrollToBottom();
  return bubble;
}

function appendTyping() {
  const msg = document.createElement("div");
  msg.className = "message ai";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "⚖️";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  const indicator = document.createElement("div");
  indicator.className = "typing-indicator";
  indicator.innerHTML = `<div class="dot"></div><div class="dot"></div><div class="dot"></div>`;
  bubble.appendChild(indicator);

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  messagesContainer.appendChild(msg);
  scrollToBottom();
  return msg;
}

function appendAIMessage(data) {
  const msg = document.createElement("div");
  msg.className = "message ai";

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = "⚖️";

  const bubble = document.createElement("div");
  bubble.className = "bubble";

  // Answer text
  const answerEl = document.createElement("div");
  answerEl.className = "answer-text";
  answerEl.textContent = data.answer;
  bubble.appendChild(answerEl);

  // Top score bar
  if (data.top_score !== null && data.top_score !== undefined && data.sufficient_context) {
    const scoreWrap = document.createElement("div");
    scoreWrap.className = "score-bar-wrap";
    const pct = Math.round(data.top_score * 100);
    scoreWrap.innerHTML = `
      <span class="score-bar-label">Relevance</span>
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width:${pct}%"></div>
      </div>
      <span class="score-value">${data.top_score.toFixed(3)}</span>
    `;
    bubble.appendChild(scoreWrap);
  }

  // Insufficient context banner
  if (!data.sufficient_context) {
    const banner = document.createElement("div");
    banner.className = "insufficient-banner";
    banner.textContent = "⚠️ Low retrieval confidence — consider rephrasing or checking SEBI's official site.";
    bubble.appendChild(banner);
  }

  // Sources
  if (data.sources && data.sources.length > 0) {
    const sec = document.createElement("div");
    sec.className = "sources-section";

    const header = document.createElement("div");
    header.className = "sources-header";
    header.innerHTML = `📖 Sources (${data.sources.length})`;
    sec.appendChild(header);

    const list = document.createElement("div");
    list.className = "sources-list";

    data.sources.forEach(src => {
      const card = document.createElement("div");
      card.className = "source-card";

      const title = document.createElement("div");
      title.className = "source-title";
      title.textContent = src.title || "SEBI Document";
      card.appendChild(title);

      const meta = document.createElement("div");
      meta.className = "source-meta";

      if (src.section_number) {
        const pill = document.createElement("span");
        pill.className = "source-pill pill-section";
        pill.textContent = `§ ${src.section_number}`;
        meta.appendChild(pill);
      }

      const forcePill = document.createElement("span");
      forcePill.className = `source-pill ${src.in_force ? "pill-inforce" : "pill-superseded"}`;
      forcePill.textContent = src.in_force ? "In Force" : "Superseded";
      meta.appendChild(forcePill);

      if (src.score !== undefined) {
        const scorePill = document.createElement("span");
        scorePill.className = "source-pill pill-score";
        scorePill.textContent = src.score.toFixed(3);
        meta.appendChild(scorePill);
      }

      if (src.source_url) {
        const link = document.createElement("a");
        link.className = "source-link";
        link.href      = src.source_url;
        link.target    = "_blank";
        link.rel       = "noopener noreferrer";
        link.textContent = "View ↗";
        meta.appendChild(link);
      }

      card.appendChild(meta);
      list.appendChild(card);
    });

    sec.appendChild(list);
    bubble.appendChild(sec);
  }

  msg.appendChild(avatar);
  msg.appendChild(bubble);
  messagesContainer.appendChild(msg);
  scrollToBottom();
}

// ── Utilities ─────────────────────────────────────────────────────────────

function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
  });
}

function showToast(msg) {
  let toast = document.querySelector(".toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "toast";
    document.body.appendChild(toast);
  }
  toast.textContent = msg;
  toast.classList.add("show");
  setTimeout(() => toast.classList.remove("show"), 4000);
}
