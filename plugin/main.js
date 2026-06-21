const { Plugin, ItemView, WorkspaceLeaf, Setting, PluginSettingTab, requestUrl, MarkdownRenderer, Modal } = require("obsidian");

const SEARCH_VIEW_TYPE = "vault-search-view";
const CHAT_VIEW_TYPE = "vault-chat-view";
const CONTEXT_VIEW_TYPE = "vault-context-view";
const DEFAULT_SETTINGS = {
  apiUrl: "http://localhost:3789",
  nResults: 20,
  useVaultContext: true,
  apiKey: "",
  chatMode: "hybrid",
  useActiveNote: false,
  excludeFolders: ".obsidian, .trash",
  contextNResults: 10,
  chatHistoryFolder: "VaultChatHistory",
};

// --- Folder filter ---
function isFileAllowed(file, settings) {
  const excludes = (settings.excludeFolders || "")
    .split(",")
    .map((s) => s.trim().toLowerCase())
    .filter(Boolean);
  const path = file.path.toLowerCase();
  return !excludes.some((ex) => path.startsWith(ex + "/") || path.startsWith(ex + "\\"));
}

// --- Helpers ---

function extractSection(content, sectionName) {
  const lines = content.split("\n");
  let capturing = false;
  let captureLevel = 0;
  const captured = [];

  for (const line of lines) {
    const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
    if (headingMatch) {
      const level = headingMatch[1].length;
      const title = headingMatch[2].trim();
      if (capturing) {
        if (level <= captureLevel) break;
        captured.push(line);
      } else if (title === sectionName) {
        capturing = true;
        captureLevel = level;
        captured.push(line);
      }
    } else if (capturing) {
      captured.push(line);
    }
  }

  return captured.join("\n").trim();
}

/**
 * Build headers object for API requests, always including API key if set.
 */
function apiHeaders(plugin, extra = {}) {
  const headers = { ...extra };
  if (plugin.settings.apiKey) {
    headers["X-API-Key"] = plugin.settings.apiKey;
  }
  return headers;
}

/**
 * Simple fuzzy match: checks if all characters in query appear in target in order.
 */
function fuzzyMatch(query, target) {
  const q = query.toLowerCase();
  const t = target.toLowerCase();
  let qi = 0;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) qi++;
  }
  return qi === q.length;
}

// --- Search View ---
class VaultSearchView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
  }

  getViewType() { return SEARCH_VIEW_TYPE; }
  getDisplayText() { return "Vault Search"; }
  getIcon() { return "search"; }

  async onOpen() {
    const container = this.contentEl;
    container.empty();
    container.addClass("vault-search-container");

    const inputRow = container.createDiv({ cls: "vault-search-input-row" });
    this.inputEl = inputRow.createEl("input", {
      cls: "vault-search-input",
      attr: { type: "text", placeholder: "搜尋 vault..." },
    });
    const btn = inputRow.createEl("button", { cls: "vault-search-btn", text: "搜尋" });

    this.resultsEl = container.createDiv({ cls: "vault-search-results" });
    this.showStatus("輸入關鍵字開始語意搜尋");

    btn.addEventListener("click", () => this.doSearch());
    this.inputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter") this.doSearch();
    });
  }

  showStatus(msg, isError = false) {
    this.resultsEl.empty();
    const el = this.resultsEl.createDiv({ cls: "vault-search-status" });
    el.setText(msg);
    if (isError) el.addClass("vault-search-error");
  }

  async doSearch() {
    const query = this.inputEl.value.trim();
    if (!query) return;

    this.showStatus("搜尋中...");

    try {
      const url = `${this.plugin.settings.apiUrl}/api/search?query=${encodeURIComponent(query)}&n_results=${this.plugin.settings.nResults}`;
      const response = await requestUrl({ url, method: "GET", headers: apiHeaders(this.plugin) });
      const data = response.json;

      if (!data.results || data.results.length === 0) {
        this.showStatus("找不到相關筆記");
        return;
      }

      this.renderResults(data.results);
    } catch (e) {
      const msg = e.message || String(e);
      if (msg.includes("ECONNREFUSED") || msg.includes("fetch") || msg.includes("net::")) {
        this.showStatus("無法連線到搜尋伺服器 — 請確認 api_server.py 正在執行", true);
      } else {
        this.showStatus(`錯誤：${msg}`, true);
      }
    }
  }

  renderResults(results) {
    this.resultsEl.empty();

    const primary = results.filter((r) => r.similarity >= 0.7);
    const secondary = results.filter((r) => r.similarity >= 0.5 && r.similarity < 0.7);

    if (primary.length > 0) {
      this.resultsEl.createDiv({ cls: "vault-search-group-heading", text: `主要相關 (${primary.length})` });
      primary.forEach((r) => this.renderOneResult(r));
    }

    if (secondary.length > 0) {
      this.resultsEl.createDiv({ cls: "vault-search-group-heading", text: `其他相關 (${secondary.length})` });
      secondary.forEach((r) => this.renderOneResult(r));
    }

    if (primary.length === 0 && secondary.length === 0) {
      this.showStatus("找不到 similarity ≥ 0.50 的結果");
    }
  }

  renderOneResult(r) {
    const el = this.resultsEl.createDiv({ cls: "vault-search-result" });

    const headerRow = el.createDiv({ cls: "vault-search-header-row" });
    const toggle = headerRow.createSpan({ cls: "vault-search-toggle", text: "▶" });
    const titleSpan = headerRow.createSpan({ cls: "vault-search-note-name", text: r.note });
    if (r.section) {
      headerRow.createSpan({ cls: "vault-search-section", text: `› ${r.section}` });
    }

    const meta = el.createDiv({ cls: "vault-search-meta" });
    meta.createSpan({ cls: "vault-search-sim", text: r.similarity.toFixed(2) });
    if (r.folder) {
      meta.createSpan({ text: r.folder });
    }

    const previewEl = el.createDiv({ cls: "vault-search-preview" });
    previewEl.style.display = "none";
    let previewLoaded = false;

    toggle.addEventListener("click", async (e) => {
      e.stopPropagation();
      const isOpen = previewEl.style.display !== "none";
      if (isOpen) {
        previewEl.style.display = "none";
        toggle.setText("▶");
      } else {
        toggle.setText("▼");
        previewEl.style.display = "block";
        if (!previewLoaded) {
          previewLoaded = true;
          await this.loadPreview(r, previewEl);
        }
      }
    });

    const openNote = () => this.openNoteAtSection(r);
    titleSpan.addEventListener("click", (e) => {
      e.stopPropagation();
      openNote();
    });
    el.addEventListener("click", () => openNote());
  }

  async loadPreview(r, containerEl) {
    const file = this.app.vault.getAbstractFileByPath(r.file)
      || this.app.vault.getMarkdownFiles().find((f) => f.basename === r.note);

    if (!file) {
      containerEl.setText("找不到檔案");
      return;
    }

    const content = await this.app.vault.cachedRead(file);
    let excerpt = "";

    if (r.section) {
      excerpt = extractSection(content, r.section);
    }

    if (!excerpt) {
      const bodyStart = content.indexOf("---", 3);
      const body = bodyStart > 0 ? content.slice(bodyStart + 3).trim() : content;
      excerpt = body.slice(0, 800);
    }

    if (excerpt.length > 1000) {
      excerpt = excerpt.slice(0, 1000) + "\n\n…";
    }

    containerEl.empty();
    await MarkdownRenderer.render(this.app, excerpt, containerEl, r.file, this);
  }

  async openNoteAtSection(r) {
    const file = this.app.vault.getAbstractFileByPath(r.file)
      || this.app.vault.getMarkdownFiles().find((f) => f.basename === r.note);

    if (!file) return;

    const leaf = this.app.workspace.getLeaf(false);
    if (r.section) {
      await leaf.openFile(file, { eState: { subpath: "#" + r.section } });
    } else {
      await leaf.openFile(file);
    }
  }
}

// --- Context View (Related Notes) ---
class VaultContextView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
    this.isPaused = false;
    this.currentNoteName = null;
    this.currentSelection = null;
    this.queryMode = "similar"; // "similar" | "search"
    this.debounceTimer = null;
    this.selectionDebounceTimer = null;
    this.requestGeneration = 0;
    this.editorListenerCleanup = null;
  }

  getViewType() { return CONTEXT_VIEW_TYPE; }
  getDisplayText() { return "Related Notes"; }
  getIcon() { return "git-compare"; }

  async onOpen() {
    const container = this.contentEl;
    container.empty();
    container.addClass("vault-search-container");

    // Toolbar
    const toolbar = container.createDiv({ cls: "vault-context-toolbar" });

    this.pauseBtn = toolbar.createEl("button", { cls: "vault-context-toolbar-btn", text: "⏸" });
    this.pauseBtn.title = "暫停自動更新";
    this.pauseBtn.addEventListener("click", () => {
      this.isPaused = !this.isPaused;
      this.pauseBtn.setText(this.isPaused ? "▶" : "⏸");
      this.pauseBtn.title = this.isPaused ? "繼續自動更新" : "暫停自動更新";
      this.pauseBtn.toggleClass("is-active", this.isPaused);
    });

    const refreshBtn = toolbar.createEl("button", { cls: "vault-context-toolbar-btn", text: "↻" });
    refreshBtn.title = "重新查詢";
    refreshBtn.addEventListener("click", () => this.fetchAndRender());

    this.reindexBtn = toolbar.createEl("button", { cls: "vault-context-toolbar-btn", text: "🔄" });
    this.reindexBtn.title = "重新索引當前筆記";
    this.reindexBtn.addEventListener("click", () => this.reindexCurrentNote());

    this.statusEl = toolbar.createSpan({ cls: "vault-context-status" });

    // Results
    this.resultsEl = container.createDiv({ cls: "vault-search-results" });

    // Listen for active note changes
    this.registerEvent(this.app.workspace.on("active-leaf-change", () => {
      this.onActiveLeafChange();
    }));

    // Initial query
    this.onActiveLeafChange();
  }

  onClose() {
    this.cleanupEditorListeners();
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    if (this.selectionDebounceTimer) clearTimeout(this.selectionDebounceTimer);
  }

  cleanupEditorListeners() {
    if (this.editorListenerCleanup) {
      this.editorListenerCleanup();
      this.editorListenerCleanup = null;
    }
  }

  onActiveLeafChange() {
    if (this.isPaused) return;

    // Skip if the active leaf is the context view itself
    const activeLeaf = this.app.workspace.activeLeaf;
    if (activeLeaf && activeLeaf.view && activeLeaf.view.getViewType &&
        activeLeaf.view.getViewType() === CONTEXT_VIEW_TYPE) return;

    this.cleanupEditorListeners();

    const file = this.app.workspace.getActiveFile();
    if (!file || !file.path.endsWith(".md")) return;

    const noteName = file.basename;

    // Attach selection listeners to the new editor
    this.attachSelectionListeners();

    // Debounce note-level query
    if (this.debounceTimer) clearTimeout(this.debounceTimer);
    this.debounceTimer = setTimeout(() => {
      this.currentNoteName = noteName;
      this.currentSelection = null;
      this.queryMode = "similar";
      this.fetchAndRender();
    }, 500);
  }

  attachSelectionListeners() {
    const activeFile = this.app.workspace.getActiveFile();
    if (!activeFile) return;

    const markdownLeaves = this.app.workspace.getLeavesOfType("markdown");
    const targetLeaf = markdownLeaves.find((l) => l.view && l.view.file && l.view.file.path === activeFile.path);
    if (!targetLeaf || !targetLeaf.view || !targetLeaf.view.editor) return;

    const editor = targetLeaf.view.editor;
    const cmDom = (editor.cm && editor.cm.dom)
      || targetLeaf.view.contentEl.querySelector(".cm-editor");
    if (!cmDom) return;

    const handler = () => {
      if (this.isPaused) return;
      const selection = editor.getSelection();

      if (selection && selection.trim().length > 10) {
        if (this.selectionDebounceTimer) clearTimeout(this.selectionDebounceTimer);
        this.selectionDebounceTimer = setTimeout(() => {
          this.currentSelection = selection.trim();
          this.queryMode = "search";
          this.fetchAndRender();
        }, 800);
      } else if (this.queryMode === "search") {
        if (this.selectionDebounceTimer) clearTimeout(this.selectionDebounceTimer);
        this.selectionDebounceTimer = setTimeout(() => {
          this.currentSelection = null;
          this.queryMode = "similar";
          this.fetchAndRender();
        }, 500);
      }
    };

    cmDom.addEventListener("mouseup", handler);
    cmDom.addEventListener("keyup", handler);

    this.editorListenerCleanup = () => {
      cmDom.removeEventListener("mouseup", handler);
      cmDom.removeEventListener("keyup", handler);
    };
  }

  updateStatus(text, state = "done") {
    this.statusEl.setText(text);
    this.statusEl.removeClass("is-loading", "is-error");
    if (state === "loading") this.statusEl.addClass("is-loading");
    if (state === "error") this.statusEl.addClass("is-error");
  }

  async fetchAndRender() {
    const generation = ++this.requestGeneration;
    const nResults = this.plugin.settings.contextNResults || 10;
    const baseUrl = this.plugin.settings.apiUrl;
    let url, statusText;

    if (this.queryMode === "search" && this.currentSelection) {
      const queryText = this.currentSelection.slice(0, 500);
      url = `${baseUrl}/api/search?query=${encodeURIComponent(queryText)}&n_results=${nResults}`;
      const display = queryText.slice(0, 40) + (queryText.length > 40 ? "..." : "");
      statusText = `搜尋：「${display}」`;
    } else if (this.currentNoteName) {
      // Try /api/similar first, fall back to /api/search if it fails
      url = `${baseUrl}/api/similar?note_name=${encodeURIComponent(this.currentNoteName)}&n_results=${nResults}`;
      this.similarFallbackQuery = this.currentNoteName;
      statusText = `相關：${this.currentNoteName}`;
    } else {
      return;
    }

    this.updateStatus(statusText, "loading");

    // Helper: fetch with one retry on connection error (server may be reloading)
    const fetchWithRetry = async (reqUrl) => {
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          return await requestUrl({ url: reqUrl, method: "GET", headers: apiHeaders(this.plugin) });
        } catch (err) {
          const m = err.message || String(err);
          if (attempt === 0 && (m.includes("ECONNREFUSED") || m.includes("net::") || m.includes("fetch") || m.includes("Failed to fetch") || m.includes("NetworkError"))) {
            await new Promise((r) => setTimeout(r, 2000));
            continue;
          }
          throw err;
        }
      }
    };

    try {
      const response = await fetchWithRetry(url);
      if (generation !== this.requestGeneration) return;
      const data = response.json;

      if (!data.results || data.results.length === 0) {
        this.updateStatus(statusText, "done");
        this.resultsEl.empty();
        this.resultsEl.createDiv({ cls: "vault-search-status", text: "找不到相關筆記" });
        return;
      }

      this.updateStatus(`${statusText} (${data.results.length})`, "done");
      this.renderResults(data.results);
    } catch (e) {
      if (generation !== this.requestGeneration) return;
      const msg = e.message || String(e);
      // Fallback: if /api/similar fails (500/404), retry with /api/search using note name
      if (this.similarFallbackQuery && (msg.includes("500") || msg.includes("404"))) {
        const fallbackName = this.similarFallbackQuery;
        this.similarFallbackQuery = null;
        try {
          const fallbackUrl = `${baseUrl}/api/search?query=${encodeURIComponent(fallbackName)}&n_results=${nResults}`;
          const resp2 = await fetchWithRetry(fallbackUrl);
          if (generation !== this.requestGeneration) return;
          const data2 = resp2.json;
          if (data2.results && data2.results.length > 0) {
            this.updateStatus(`${statusText} (${data2.results.length})`, "done");
            this.renderResults(data2.results);
            return;
          }
        } catch (_) { /* fall through to error display */ }
      }
      if (msg.includes("ECONNREFUSED") || msg.includes("fetch") || msg.includes("net::")) {
        this.updateStatus("無法連線", "error");
      } else {
        this.updateStatus(statusText, "done");
        this.resultsEl.empty();
        this.resultsEl.createDiv({ cls: "vault-search-status", text: "筆記尚未索引 — 按 🔄 索引" });
      }
    }
  }

  renderResults(results) {
    this.resultsEl.empty();

    const primary = results.filter((r) => r.similarity >= 0.7);
    const secondary = results.filter((r) => r.similarity >= 0.5 && r.similarity < 0.7);

    if (primary.length > 0) {
      this.resultsEl.createDiv({ cls: "vault-search-group-heading", text: `主要相關 (${primary.length})` });
      primary.forEach((r) => this.renderOneResult(r));
    }

    if (secondary.length > 0) {
      this.resultsEl.createDiv({ cls: "vault-search-group-heading", text: `其他相關 (${secondary.length})` });
      secondary.forEach((r) => this.renderOneResult(r));
    }

    if (primary.length === 0 && secondary.length === 0) {
      this.resultsEl.createDiv({ cls: "vault-search-status", text: "找不到 similarity ≥ 0.50 的結果" });
    }
  }

  renderOneResult(r) {
    const el = this.resultsEl.createDiv({ cls: "vault-search-result" });

    const headerRow = el.createDiv({ cls: "vault-search-header-row" });
    const toggle = headerRow.createSpan({ cls: "vault-search-toggle", text: "▶" });
    const titleSpan = headerRow.createSpan({ cls: "vault-search-note-name", text: r.note });
    if (r.section) {
      headerRow.createSpan({ cls: "vault-search-section", text: `› ${r.section}` });
    }

    const meta = el.createDiv({ cls: "vault-search-meta" });
    meta.createSpan({ cls: "vault-search-sim", text: r.similarity.toFixed(2) });
    if (r.folder) {
      meta.createSpan({ text: r.folder });
    }

    const previewEl = el.createDiv({ cls: "vault-search-preview" });
    previewEl.style.display = "none";
    let previewLoaded = false;

    toggle.addEventListener("click", async (e) => {
      e.stopPropagation();
      const isOpen = previewEl.style.display !== "none";
      if (isOpen) {
        previewEl.style.display = "none";
        toggle.setText("▶");
      } else {
        toggle.setText("▼");
        previewEl.style.display = "block";
        if (!previewLoaded) {
          previewLoaded = true;
          const file = this.app.vault.getAbstractFileByPath(r.file)
            || this.app.vault.getMarkdownFiles().find((f) => f.basename === r.note);
          if (!file) { previewEl.setText("找不到檔案"); return; }
          const content = await this.app.vault.cachedRead(file);
          let excerpt = r.section ? extractSection(content, r.section) : "";
          if (!excerpt) {
            const bodyStart = content.indexOf("---", 3);
            const body = bodyStart > 0 ? content.slice(bodyStart + 3).trim() : content;
            excerpt = body.slice(0, 800);
          }
          if (excerpt.length > 1000) excerpt = excerpt.slice(0, 1000) + "\n\n…";
          previewEl.empty();
          await MarkdownRenderer.render(this.app, excerpt, previewEl, r.file, this);
        }
      }
    });

    const openNote = async () => {
      const file = this.app.vault.getAbstractFileByPath(r.file)
        || this.app.vault.getMarkdownFiles().find((f) => f.basename === r.note);
      if (!file) return;
      const leaf = this.app.workspace.getLeaf(false);
      if (r.section) {
        await leaf.openFile(file, { eState: { subpath: "#" + r.section } });
      } else {
        await leaf.openFile(file);
      }
    };
    titleSpan.addEventListener("click", (e) => { e.stopPropagation(); openNote(); });
    el.addEventListener("click", () => openNote());
  }

  async reindexCurrentNote() {
    const file = this.app.workspace.getActiveFile();
    if (!file) return;

    this.reindexBtn.disabled = true;
    this.updateStatus(`索引中：${file.basename}...`, "loading");

    try {
      const url = `${this.plugin.settings.apiUrl}/api/reindex`;
      const response = await requestUrl({
        url,
        method: "POST",
        headers: apiHeaders(this.plugin, { "Content-Type": "application/json" }),
        body: JSON.stringify({ file: file.path }),
      });
      const data = response.json;
      this.updateStatus(`已索引：${data.note} (${data.chunks} chunks)`, "done");
      // Refresh results after reindex
      setTimeout(() => this.fetchAndRender(), 300);
    } catch (e) {
      this.updateStatus(`索引失敗：${e.message || e}`, "error");
    } finally {
      this.reindexBtn.disabled = false;
    }
  }
}

// --- Context Preview Modal ---
class ContextPreviewModal extends Modal {
  constructor(app, contextData) {
    super(app);
    this.contextData = contextData;
    this.confirmed = false;
  }

  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.createEl("h3", { text: "👁 上下文預覽" });

    const { activeNote, mentionedNotes, selectedNotes, estimatedTokens } = this.contextData;

    // Active note
    if (activeNote) {
      const section = contentEl.createDiv({ cls: "vault-chat-preview-section" });
      section.createEl("h4", { text: `📄 開啟頁面 (${activeNote.type})` });
      section.createEl("div", { cls: "vault-chat-preview-file", text: activeNote.file });
      const preview = section.createEl("pre", { cls: "vault-chat-preview-content" });
      preview.setText(activeNote.content.slice(0, 500) + (activeNote.content.length > 500 ? "..." : ""));
    }

    // Mentioned notes
    if (mentionedNotes && mentionedNotes.length > 0) {
      const section = contentEl.createDiv({ cls: "vault-chat-preview-section" });
      section.createEl("h4", { text: `@ 提及筆記 (${mentionedNotes.length})` });
      mentionedNotes.forEach((n) => {
        section.createEl("div", { cls: "vault-chat-preview-file", text: n });
      });
    }

    // Selected search notes
    if (selectedNotes && selectedNotes.length > 0) {
      const section = contentEl.createDiv({ cls: "vault-chat-preview-section" });
      section.createEl("h4", { text: `🔍 搜尋選取筆記 (${selectedNotes.length})` });
      selectedNotes.forEach((n) => {
        section.createEl("div", { cls: "vault-chat-preview-file", text: n });
      });
    }

    // No context
    if (!activeNote && (!mentionedNotes || mentionedNotes.length === 0) && (!selectedNotes || selectedNotes.length === 0)) {
      contentEl.createEl("div", { cls: "vault-chat-preview-empty", text: "沒有附加上下文" });
    }

    // Token estimate
    const tokenEl = contentEl.createDiv({ cls: "vault-chat-preview-tokens" });
    tokenEl.setText(`預估 tokens: ~${Math.round(estimatedTokens).toLocaleString()}`);

    // Buttons
    const btnRow = contentEl.createDiv({ cls: "vault-chat-preview-btns" });
    const confirmBtn = btnRow.createEl("button", { cls: "mod-cta", text: "確認送出" });
    const cancelBtn = btnRow.createEl("button", { text: "取消" });

    confirmBtn.addEventListener("click", () => {
      this.confirmed = true;
      this.close();
    });
    cancelBtn.addEventListener("click", () => {
      this.confirmed = false;
      this.close();
    });
  }

  onClose() {
    this.contentEl.empty();
    if (this.resolvePromise) {
      this.resolvePromise(this.confirmed);
    }
  }

  waitForResult() {
    return new Promise((resolve) => {
      this.resolvePromise = resolve;
    });
  }
}

// --- Chat View ---
class VaultChatView extends ItemView {
  constructor(leaf, plugin) {
    super(leaf);
    this.plugin = plugin;
    this.history = [];
    this.isWaiting = false;
    this.selectedNotes = new Map();
    this.totalUsage = { input: 0, output: 0, cost: 0 };
    this.chatMode = plugin.settings.chatMode || "hybrid";
    this.useActiveNote = plugin.settings.useActiveNote || false;
    this.mentionedNotes = [];
    this.rawMarkdownMap = new WeakMap();
    this.inputHistory = [];
    this.inputHistoryIndex = -1;
  }

  getViewType() { return CHAT_VIEW_TYPE; }
  getDisplayText() { return "Vault Chat"; }
  getIcon() { return "message-circle"; }

  // Detect if cursor is on the first visual line (accounts for soft-wrap)
  _isCursorOnFirstVisualLine() {
    const ta = this.chatInput;
    const pos = ta.selectionStart;
    if (pos === 0) return true;
    const textBefore = ta.value.substring(0, pos);
    if (textBefore.includes("\n")) return false;
    // Soft-wrap detection: measure text height in a mirror element
    if (!this._mirror) {
      this._mirror = document.createElement("div");
      this._mirror.style.cssText = "position:fixed;left:-9999px;top:-9999px;white-space:pre-wrap;word-wrap:break-word;overflow-wrap:break-word;";
      document.body.appendChild(this._mirror);
    }
    const style = getComputedStyle(ta);
    this._mirror.style.width = style.width;
    this._mirror.style.fontSize = style.fontSize;
    this._mirror.style.fontFamily = style.fontFamily;
    this._mirror.style.fontWeight = style.fontWeight;
    this._mirror.style.letterSpacing = style.letterSpacing;
    this._mirror.style.padding = style.padding;
    this._mirror.style.border = style.border;
    this._mirror.style.boxSizing = style.boxSizing;
    this._mirror.textContent = textBefore;
    const lineHeight = parseFloat(style.lineHeight) || parseFloat(style.fontSize) * 1.2 || 20;
    return this._mirror.offsetHeight <= lineHeight + 2;
  }

  async onOpen() {
    const container = this.contentEl;
    container.empty();
    container.addClass("vault-chat-container");

    // === Toolbar ===
    const toolbar = container.createDiv({ cls: "vault-chat-toolbar" });

    // Mode switcher button group
    const modeGroup = toolbar.createDiv({ cls: "vault-chat-mode-group" });
    this.modeBtns = {};

    const modes = [
      { key: "vault", label: "📚", title: "Vault — only use vault notes" },
      { key: "free", label: "💬", title: "Free — no vault search" },
      { key: "hybrid", label: "🔀", title: "Hybrid — auto-decide" },
    ];

    modes.forEach(({ key, label, title }) => {
      const btn = modeGroup.createEl("button", {
        cls: "vault-chat-mode-btn",
        text: label,
        attr: { title },
      });
      if (key === this.chatMode) btn.addClass("vault-chat-mode-active");
      btn.addEventListener("click", () => {
        this.chatMode = key;
        Object.values(this.modeBtns).forEach((b) => b.removeClass("vault-chat-mode-active"));
        btn.addClass("vault-chat-mode-active");
        this.showModeHint();
      });
      this.modeBtns[key] = btn;
    });

    // Context group: active note + preview
    const contextGroup = toolbar.createDiv({ cls: "vault-chat-context-group" });

    this.activeNoteBtn = contextGroup.createEl("button", {
      cls: "vault-chat-toolbar-icon",
      text: "📄",
      attr: { title: "使用開啟頁面" },
    });
    this.activeNoteBtn.addEventListener("click", () => {
      this.useActiveNote = !this.useActiveNote;
      if (this.useActiveNote) {
        this.activeNoteBtn.classList.add("vault-chat-toolbar-icon-active");
      } else {
        this.activeNoteBtn.classList.remove("vault-chat-toolbar-icon-active");
      }
    });
    if (this.useActiveNote) this.activeNoteBtn.classList.add("vault-chat-toolbar-icon-active");
    // Hidden checkbox for compatibility
    this.activeNoteCb = createEl("input", { attr: { type: "checkbox" } });
    this.activeNoteCb.checked = this.useActiveNote;
    this.activeNoteInfo = createEl("span");

    const previewBtnToolbar = contextGroup.createEl("button", {
      cls: "vault-chat-toolbar-icon",
      text: "👁",
      attr: { title: "預覽 context" },
    });
    previewBtnToolbar.addEventListener("click", () => this.showContextPreview());

    // Action group: clear + save
    const actionGroup = toolbar.createDiv({ cls: "vault-chat-action-group" });

    const clearBtn = actionGroup.createEl("button", { cls: "vault-chat-toolbar-icon", text: "🗑", attr: { title: "清除對話" } });
    clearBtn.addEventListener("click", () => {
      this.history = [];
      this.totalUsage = { input: 0, output: 0, cost: 0 };
      this.mentionedNotes = [];
      this.selectedNotes.clear();
      this.messagesEl.empty();
      this.updateUsageDisplay();
      this.updateMentionChips();
      this.addSystemMessage("對話已清除。新對話開始。");
    });

    const saveBtn = actionGroup.createEl("button", { cls: "vault-chat-toolbar-icon", text: "💾", attr: { title: "儲存對話" } });
    saveBtn.addEventListener("click", () => this.saveConversation());

    // Usage display removed — was pushing content off-screen on mobile

    // Listen for active file changes to update info
    this.registerEvent(this.app.workspace.on("active-leaf-change", () => {
      if (this.useActiveNote) this.updateActiveNoteInfo();
    }));

    // === Messages area ===
    this.messagesEl = container.createDiv({ cls: "vault-chat-messages" });
    this.showModeHint();

    // Context picker area (hidden by default)
    this.contextPickerEl = container.createDiv({ cls: "vault-chat-context-picker" });
    this.contextPickerEl.style.display = "none";

    // === Mentioned notes chips (P2b) ===
    this.mentionChipsEl = container.createDiv({ cls: "vault-chat-mention-chips" });

    // === Input area ===
    const inputArea = container.createDiv({ cls: "vault-chat-input-area" });

    // Input row (textarea + buttons on the same line)
    const inputRow = inputArea.createDiv({ cls: "vault-chat-input-row-main" });

    // Textarea wrapper (for autocomplete positioning)
    const textareaWrapper = inputRow.createDiv({ cls: "vault-chat-textarea-wrapper" });
    this.chatInput = textareaWrapper.createEl("textarea", {
      cls: "vault-chat-input",
      attr: { placeholder: "訊息... (@ 提及筆記)", rows: "2" },
    });

    // Autocomplete dropdown (P2b)
    this.autocompleteEl = textareaWrapper.createDiv({ cls: "vault-chat-autocomplete" });
    this.autocompleteEl.style.display = "none";
    this.setupMentionAutocomplete();

    // Icon buttons beside textarea
    const btnCol = inputRow.createDiv({ cls: "vault-chat-btn-col" });
    const searchCtxBtn = btnCol.createEl("button", { cls: "vault-chat-icon-btn", text: "📎", attr: { title: "選擇參考筆記" } });
    const sendBtn = btnCol.createEl("button", { cls: "vault-chat-send-icon-btn", text: "➤", attr: { title: "送出" } });

    searchCtxBtn.addEventListener("click", () => this.searchContext());
    sendBtn.addEventListener("click", () => this.sendMessage());
    this.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        // Don't send if autocomplete is visible or just closed
        if (this.autocompleteEl.style.display !== "none" || this._justSelectedAutocomplete) {
          return;
        }
        e.preventDefault();
        this.sendMessage();
      }
      // Input history: up/down arrows
      if (e.key === "ArrowUp" && this.autocompleteEl.style.display === "none") {
        if (this._isCursorOnFirstVisualLine() && this.inputHistory.length > 0) {
          e.preventDefault();
          if (this.inputHistoryIndex === -1) {
            this.inputHistoryDraft = this.chatInput.value;
            this.inputHistoryIndex = this.inputHistory.length - 1;
          } else if (this.inputHistoryIndex > 0) {
            this.inputHistoryIndex--;
          }
          this.chatInput.value = this.inputHistory[this.inputHistoryIndex];
          this.chatInput.selectionStart = this.chatInput.selectionEnd = this.chatInput.value.length;
        }
      }
      if (e.key === "ArrowDown" && this.autocompleteEl.style.display === "none") {
        if (this.inputHistoryIndex !== -1) {
          e.preventDefault();
          if (this.inputHistoryIndex < this.inputHistory.length - 1) {
            this.inputHistoryIndex++;
            this.chatInput.value = this.inputHistory[this.inputHistoryIndex];
          } else {
            this.inputHistoryIndex = -1;
            this.chatInput.value = this.inputHistoryDraft || "";
          }
        }
      }
    });
  }

  // --- Active note info (P2a) ---

  updateActiveNoteInfo() {
    this.activeNoteInfo.empty();
    if (!this.useActiveNote) return;
    const file = this.app.workspace.getActiveFile();
    if (file) {
      this.activeNoteInfo.setText(file.basename);
    } else {
      this.activeNoteInfo.setText("(無開啟檔案)");
    }
  }

  getActiveNoteContext() {
    if (!this.useActiveNote) return null;

    const file = this.app.workspace.getActiveFile();
    if (!file) return null;

    // Find the markdown leaf that has this file open (not the sidebar chat leaf)
    const markdownLeaves = this.app.workspace.getLeavesOfType("markdown");
    const targetLeaf = markdownLeaves.find((l) => l.view?.file?.path === file.path);
    const editor = targetLeaf?.view?.editor;

    if (editor) {
      // Check for selection
      const selection = editor.getSelection();
      if (selection && selection.trim().length > 0) {
        return { file: file.path, content: selection, type: "selection" };
      }

      // Check if cursor is on a heading line
      const cursor = editor.getCursor();
      const line = editor.getLine(cursor.line);
      const headingMatch = line.match(/^(#{1,6})\s+(.*)$/);
      if (headingMatch) {
        const fullContent = editor.getValue();
        const sectionContent = extractSection(fullContent, headingMatch[2].trim());
        if (sectionContent) {
          return { file: file.path, content: sectionContent, type: "section" };
        }
      }
    }

    // Fallback: full file (will be read async)
    return { file: file.path, content: null, type: "full" };
  }

  async resolveActiveNoteContent(ctx) {
    if (!ctx) return null;
    if (ctx.content !== null) return ctx;

    // Read full file content
    const file = this.app.vault.getAbstractFileByPath(ctx.file);
    if (!file) return null;
    const content = await this.app.vault.cachedRead(file);
    const truncated = content.length > 5000 ? content.slice(0, 5000) : content;
    return { ...ctx, content: truncated };
  }

  // --- @ Mention autocomplete (P2b) ---

  setupMentionAutocomplete() {
    this.mentionQuery = null;
    this.mentionStart = -1;
    this.autocompleteIndex = 0;

    this.chatInput.addEventListener("input", () => {
      this.handleMentionInput();
    });

    this.chatInput.addEventListener("keydown", (e) => {
      if (this.autocompleteEl.style.display === "none") return;

      if (e.key === "ArrowDown") {
        e.preventDefault();
        this.navigateAutocomplete(1);
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        this.navigateAutocomplete(-1);
      } else if (e.key === "Enter" || e.key === "Tab") {
        e.preventDefault();
        this.selectAutocompleteItem();
      } else if (e.key === "Escape") {
        e.preventDefault();
        this.hideAutocomplete();
      }
    });
  }

  handleMentionInput() {
    const value = this.chatInput.value;
    const cursorPos = this.chatInput.selectionStart;

    // Find @ before cursor
    const textBeforeCursor = value.slice(0, cursorPos);
    const atIndex = textBeforeCursor.lastIndexOf("@");

    if (atIndex === -1 || (atIndex > 0 && value[atIndex - 1] !== " " && value[atIndex - 1] !== "\n")) {
      this.hideAutocomplete();
      return;
    }

    const query = textBeforeCursor.slice(atIndex + 1);
    // If there's a space in the query after initial text, might be done
    if (query.includes("\n")) {
      this.hideAutocomplete();
      return;
    }

    this.mentionStart = atIndex;
    this.mentionQuery = query;
    this.showAutocomplete(query);
  }

  showAutocomplete(query) {
    const files = this.app.vault.getMarkdownFiles().filter((f) => isFileAllowed(f, this.plugin.settings));
    let matches;

    if (query.length === 0) {
      // Show recent files
      matches = files.slice(0, 10);
    } else {
      const q = query.toLowerCase();
      // Score: 0=exact, 1=startsWith, 2=contains, 3=fuzzy
      const scored = [];
      for (const f of files) {
        const name = f.basename.toLowerCase();
        if (name === q) {
          scored.push({ file: f, score: 0 });
        } else if (name.startsWith(q)) {
          scored.push({ file: f, score: 1 });
        } else if (name.includes(q)) {
          scored.push({ file: f, score: 2 });
        } else if (fuzzyMatch(query, f.basename)) {
          scored.push({ file: f, score: 3 });
        }
      }
      scored.sort((a, b) => a.score - b.score || a.file.basename.length - b.file.basename.length);
      matches = scored.slice(0, 10).map((s) => s.file);
    }

    if (matches.length === 0) {
      this.hideAutocomplete();
      return;
    }

    this.autocompleteEl.empty();
    this.autocompleteEl.style.display = "block";
    this.autocompleteIndex = 0;
    this.autocompleteItems = matches;

    matches.forEach((file, idx) => {
      const item = this.autocompleteEl.createDiv({ cls: "vault-chat-autocomplete-item" });
      item.setText(file.basename);
      if (idx === 0) item.addClass("vault-chat-autocomplete-active");
      item.addEventListener("click", () => {
        this.autocompleteIndex = idx;
        this.selectAutocompleteItem();
      });
      item.addEventListener("mouseenter", () => {
        this.autocompleteEl.querySelectorAll(".vault-chat-autocomplete-item").forEach((el) =>
          el.removeClass("vault-chat-autocomplete-active")
        );
        item.addClass("vault-chat-autocomplete-active");
        this.autocompleteIndex = idx;
      });
    });
  }

  navigateAutocomplete(direction) {
    const items = this.autocompleteEl.querySelectorAll(".vault-chat-autocomplete-item");
    if (items.length === 0) return;
    items[this.autocompleteIndex]?.removeClass("vault-chat-autocomplete-active");
    this.autocompleteIndex = (this.autocompleteIndex + direction + items.length) % items.length;
    items[this.autocompleteIndex]?.addClass("vault-chat-autocomplete-active");
  }

  selectAutocompleteItem() {
    if (!this.autocompleteItems || this.autocompleteItems.length === 0) return;
    const file = this.autocompleteItems[this.autocompleteIndex];
    if (!file) return;

    const name = file.basename;

    // Replace @query with note name in textarea
    const value = this.chatInput.value;
    const cursorPos = this.chatInput.selectionStart;
    const before = value.slice(0, this.mentionStart);
    const after = value.slice(cursorPos);
    this.chatInput.value = before + "@" + name + " " + after;
    this.chatInput.selectionStart = this.chatInput.selectionEnd = this.mentionStart + name.length + 2;

    // Add to mentioned notes
    if (!this.mentionedNotes.includes(name)) {
      this.mentionedNotes.push(name);
      this.updateMentionChips();
    }

    this.hideAutocomplete();
    this._justSelectedAutocomplete = true;
    setTimeout(() => { this._justSelectedAutocomplete = false; }, 200);
    this.chatInput.focus();
  }

  hideAutocomplete() {
    this.autocompleteEl.style.display = "none";
    this.autocompleteItems = [];
  }

  updateMentionChips() {
    this.mentionChipsEl.empty();
    if (this.mentionedNotes.length === 0) {
      this.mentionChipsEl.style.display = "none";
      return;
    }
    this.mentionChipsEl.style.display = "flex";
    this.mentionedNotes.forEach((name, idx) => {
      const chip = this.mentionChipsEl.createSpan({ cls: "vault-chat-mention-chip" });
      chip.createSpan({ text: "@" + name });
      const removeBtn = chip.createSpan({ cls: "vault-chat-mention-remove", text: " ✕" });
      removeBtn.addEventListener("click", () => {
        this.mentionedNotes.splice(idx, 1);
        this.updateMentionChips();
      });
    });
  }

  // --- Context Preview (P2c) ---

  async showContextPreview() {
    const activeNoteCtx = await this.resolveActiveNoteContent(this.getActiveNoteContext());
    const selectedNoteNames = [];
    this.selectedNotes.forEach((entry, name) => {
      if (entry.selected) selectedNoteNames.push(name);
    });

    let totalChars = 0;
    if (activeNoteCtx) totalChars += activeNoteCtx.content.length;
    // Estimate ~2000 chars per mentioned/selected note
    totalChars += this.mentionedNotes.length * 2000;
    totalChars += selectedNoteNames.length * 2000;
    totalChars += (this.chatInput.value || "").length;

    const modal = new ContextPreviewModal(this.app, {
      activeNote: activeNoteCtx,
      mentionedNotes: [...this.mentionedNotes],
      selectedNotes: selectedNoteNames,
      estimatedTokens: totalChars * 1.2,
    });
    modal.open();

    const confirmed = await modal.waitForResult();
    if (confirmed) {
      this.sendMessage();
    }
  }

  // --- Usage display (toolbar total removed, per-message usage kept) ---

  updateUsageDisplay() {
    // No-op: toolbar usage display removed for mobile usability
  }

  // --- Context picker (search notes) ---

  async searchContext() {
    const query = this.chatInput.value.trim();
    if (!query) {
      this.addSystemMessage("請先輸入問題，再點「選擇參考筆記」");
      return;
    }

    this.contextPickerEl.empty();
    this.contextPickerEl.style.display = "block";
    this.contextPickerEl.setText("搜尋中...");

    try {
      const url = `${this.plugin.settings.apiUrl}/api/search?query=${encodeURIComponent(query)}&n_results=10`;
      const response = await requestUrl({ url, method: "GET", headers: apiHeaders(this.plugin) });
      const data = response.json;

      this.contextPickerEl.empty();

      if (!data.results || data.results.length === 0) {
        this.contextPickerEl.setText("找不到相關筆記");
        return;
      }

      const header = this.contextPickerEl.createDiv({ cls: "vault-chat-picker-header" });
      header.createSpan({ text: "勾選要引用的筆記：" });
      const closeBtn = header.createSpan({ cls: "vault-chat-picker-close", text: "✕" });
      closeBtn.addEventListener("click", () => {
        this.contextPickerEl.style.display = "none";
      });

      const listEl = this.contextPickerEl.createDiv({ cls: "vault-chat-picker-list" });

      this.selectedNotes.clear();

      data.results
        .filter((r) => r.similarity >= 0.5)
        .forEach((r) => {
          const isHighSim = r.similarity >= 0.7;
          this.selectedNotes.set(r.note, { ...r, selected: isHighSim });

          const row = listEl.createDiv({ cls: "vault-chat-picker-row" });
          const cb = row.createEl("input", {
            attr: { type: "checkbox" },
            cls: "vault-chat-picker-cb",
          });
          cb.checked = isHighSim;
          cb.addEventListener("change", () => {
            const entry = this.selectedNotes.get(r.note);
            if (entry) entry.selected = cb.checked;
          });

          const label = row.createDiv({ cls: "vault-chat-picker-label" });
          label.createSpan({ cls: "vault-chat-picker-name", text: r.note });
          if (r.section) {
            label.createSpan({ cls: "vault-chat-picker-section", text: ` › ${r.section}` });
          }
          label.createSpan({ cls: "vault-chat-picker-sim", text: ` ${r.similarity.toFixed(2)}` });
        });
    } catch (e) {
      this.contextPickerEl.setText("搜尋失敗");
    }
  }

  // --- Messages ---

  showModeHint() {
    const hints = {
      vault: "📚 Vault 模式 — 僅根據 vault 筆記回答，@ 提及筆記加入 context",
      free: "💬 Free 模式 — 自由對話 + PubMed 文獻搜尋，不搜尋 vault",
      hybrid: "🔀 Hybrid 模式 — 自動搜尋 vault，不足時用 AI 知識補充",
    };
    // Remove previous hint if exists
    const prev = this.messagesEl.querySelector(".vault-chat-mode-hint");
    if (prev) prev.remove();
    const el = this.messagesEl.createDiv({ cls: "vault-chat-msg vault-chat-system vault-chat-mode-hint" });
    el.setText(hints[this.chatMode] || hints.hybrid);
    this.scrollToBottom();
  }

  addSystemMessage(text) {
    const el = this.messagesEl.createDiv({ cls: "vault-chat-msg vault-chat-system" });
    el.setText(text);
    this.scrollToBottom();
  }

  addUserMessage(text) {
    const el = this.messagesEl.createDiv({ cls: "vault-chat-msg vault-chat-user" });
    el.setText(text);
    this.scrollToBottom();
  }

  addAssistantMessage(markdown, contextNotes, usage, activeNoteCtx) {
    const wrapper = this.messagesEl.createDiv({ cls: "vault-chat-msg vault-chat-assistant" });

    // Store raw markdown for copy (P3b)
    this.rawMarkdownMap.set(wrapper, markdown);

    // Copy button (P3b)
    const copyBtn = wrapper.createDiv({ cls: "vault-chat-copy-btn", text: "📋" });
    copyBtn.setAttribute("title", "複製回覆");
    copyBtn.addEventListener("click", async () => {
      const raw = this.rawMarkdownMap.get(wrapper) || "";
      await navigator.clipboard.writeText(raw);
      copyBtn.setText("✓");
      setTimeout(() => copyBtn.setText("📋"), 1500);
    });

    // Render markdown content
    const contentEl = wrapper.createDiv({ cls: "vault-chat-assistant-content" });
    MarkdownRenderer.render(this.app, markdown, contentEl, "", this);

    // Make [[internal links]] clickable
    contentEl.querySelectorAll("a.internal-link").forEach((link) => {
      link.addEventListener("click", (e) => {
        e.preventDefault();
        const href = link.getAttribute("data-href") || link.getAttribute("href");
        if (href) {
          this.app.workspace.openLinkText(href, "", false);
        }
      });
    });

    // Suggested notes: parse [[NoteName]] from response, show as +add buttons
    const suggestedNames = [...new Set([...markdown.matchAll(/\[\[([^\]|]+?)(?:\|[^\]]*?)?\]\]/g)].map((m) => m[1]))];
    // Filter: only show notes that exist in vault and aren't already in context
    const contextSet = new Set(contextNotes || []);
    const vaultFiles = this.app.vault.getMarkdownFiles();
    const validSuggestions = suggestedNames.filter((name) => {
      if (contextSet.has(name)) return false;
      return vaultFiles.some((f) => f.basename === name);
    });

    if (validSuggestions.length > 0) {
      const suggestBar = wrapper.createDiv({ cls: "vault-chat-suggest-bar" });
      suggestBar.createSpan({ cls: "vault-chat-suggest-label", text: "建議加入：" });
      validSuggestions.forEach((name) => {
        const chip = suggestBar.createSpan({ cls: "vault-chat-suggest-chip", text: `+ ${name}` });
        chip.addEventListener("click", () => {
          if (!this.mentionedNotes.includes(name)) {
            this.mentionedNotes.push(name);
            this.updateMentionChips();
          }
          chip.setText(`✓ ${name}`);
          chip.addClass("vault-chat-suggest-added");
          chip.style.pointerEvents = "none";
        });
      });
    }

    // Footer: refs + per-message usage
    const footerEl = wrapper.createDiv({ cls: "vault-chat-msg-footer" });

    if (contextNotes && contextNotes.length > 0) {
      const refEl = footerEl.createDiv({ cls: "vault-chat-refs" });
      const refToggle = refEl.createSpan({ cls: "vault-chat-refs-toggle", text: `參考 (${contextNotes.length}) ▶` });
      const refChips = refEl.createDiv({ cls: "vault-chat-refs-chips" });
      refChips.style.display = "none";
      refToggle.addEventListener("click", () => {
        const open = refChips.style.display !== "none";
        refChips.style.display = open ? "none" : "flex";
        refToggle.setText(`參考 (${contextNotes.length}) ${open ? "▶" : "▼"}`);
      });
      contextNotes.forEach((noteName) => {
        const chip = refChips.createSpan({ cls: "vault-chat-ref-chip", text: noteName });
        chip.addEventListener("click", () => {
          const file = this.app.vault.getMarkdownFiles().find((f) => f.basename === noteName);
          if (file) {
            this.app.workspace.getLeaf(false).openFile(file);
          }
        });
      });
    }

    if (usage && (usage.input_tokens || usage.output_tokens)) {
      const usageEl = footerEl.createDiv({ cls: "vault-chat-msg-usage" });
      const inTk = usage.input_tokens || 0;
      const outTk = usage.output_tokens || 0;
      const cost = usage.cost_usd || 0;
      usageEl.setText(`${inTk.toLocaleString()} in / ${outTk.toLocaleString()} out · $${cost.toFixed(4)}`);
    }

    this.scrollToBottom();
  }

  addLoadingIndicator() {
    const el = this.messagesEl.createDiv({ cls: "vault-chat-msg vault-chat-assistant vault-chat-loading" });
    el.createSpan({ text: "思考中... " });
    const stopBtn = el.createEl("button", { cls: "vault-chat-stop-btn", text: "⏹ 停止" });
    stopBtn.addEventListener("click", () => {
      if (this.abortController) {
        this.abortController.abort();
        this.abortController = null;
      }
      // Kill server-side Claude process
      if (this._currentRequestId) {
        requestUrl({
          url: `${this.plugin.settings.apiUrl}/api/chat/cancel`,
          method: "POST",
          headers: apiHeaders(this.plugin, { "Content-Type": "application/json" }),
          body: JSON.stringify({ request_id: this._currentRequestId }),
        }).catch(() => {});
        this._currentRequestId = null;
      }
    });
    this.scrollToBottom();
    return el;
  }

  scrollToBottom() {
    this.messagesEl.scrollTop = this.messagesEl.scrollHeight;
  }

  // --- Send message ---

  async sendMessage() {
    if (this.isWaiting) return;

    const text = this.chatInput.value.trim();
    if (!text) return;

    // Save to input history
    if (this.inputHistory[this.inputHistory.length - 1] !== text) {
      this.inputHistory.push(text);
      if (this.inputHistory.length > 50) this.inputHistory.shift();
    }
    this.inputHistoryIndex = -1;

    // Gather selected notes
    const selectedNoteNames = [];
    this.selectedNotes.forEach((entry, name) => {
      if (entry.selected) selectedNoteNames.push(name);
    });

    // Get active note context (P2a)
    let activeNoteCtx = this.getActiveNoteContext();
    activeNoteCtx = await this.resolveActiveNoteContent(activeNoteCtx);

    // Copy mentioned notes before clearing
    const currentMentionedNotes = [...this.mentionedNotes];

    this.chatInput.value = "";
    this.contextPickerEl.style.display = "none";
    this.addUserMessage(text);

    this.isWaiting = true;
    this.abortController = new AbortController();
    const requestId = Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    this._currentRequestId = requestId;
    const loadingEl = this.addLoadingIndicator();

    try {
      const body = {
        message: text,
        history: this.history,
        request_id: requestId,
        mode: this.chatMode,
        n_context: 5,
      };

      // Determine use_vault_context based on chat mode
      if (this.chatMode === "vault") {
        body.use_vault_context = true;
      } else if (this.chatMode === "free") {
        body.use_vault_context = false;
      } else {
        // hybrid: use vault context by default unless user selected notes
        body.use_vault_context = this.plugin.settings.useVaultContext;
      }

      // If user manually selected notes, use those instead of auto-search
      if (selectedNoteNames.length > 0) {
        body.selected_notes = selectedNoteNames;
        body.use_vault_context = false;
      }

      // Active note context (P2a)
      if (activeNoteCtx) {
        body.active_note = {
          file: activeNoteCtx.file,
          content: activeNoteCtx.content,
          type: activeNoteCtx.type,
        };
      }

      // Mentioned notes (P2b)
      if (currentMentionedNotes.length > 0) {
        body.mentioned_notes = currentMentionedNotes;
      }

      // --- Async polling: submit job, then poll for result ---
      // Retry once on connection error (server may be reloading)
      let submitData;
      for (let attempt = 0; attempt < 2; attempt++) {
        try {
          const submitResponse = await requestUrl({
            url: `${this.plugin.settings.apiUrl}/api/chat`,
            method: "POST",
            headers: apiHeaders(this.plugin, {
              "Content-Type": "application/json",
              "X-Async": "1",
            }),
            body: JSON.stringify(body),
          });
          submitData = submitResponse.json;
          break;
        } catch (retryErr) {
          if (attempt === 0) {
            await new Promise((r) => setTimeout(r, 2000));
            continue;
          }
          throw retryErr;
        }
      }
      if (submitData.error) {
        loadingEl.remove();
        this.addSystemMessage(`錯誤：${submitData.error}`);
        return;
      }

      const pollId = submitData.request_id || requestId;

      // Poll for result every 3 seconds
      const pollResult = await new Promise((resolve, reject) => {
        const pollInterval = setInterval(async () => {
          // Check if aborted
          if (!this.abortController || this.abortController.signal.aborted) {
            clearInterval(pollInterval);
            reject(new DOMException("Aborted", "AbortError"));
            return;
          }
          try {
            const pollResponse = await requestUrl({
              url: `${this.plugin.settings.apiUrl}/api/chat/poll?request_id=${encodeURIComponent(pollId)}`,
              method: "GET",
              headers: apiHeaders(this.plugin),
            });
            const pollData = pollResponse.json;
            if (pollData.status === "processing") return; // keep polling
            clearInterval(pollInterval);
            resolve(pollData);
          } catch (pollErr) {
            // Network error during poll — keep trying
          }
        }, 3000);

        // Also listen for abort to stop polling
        this.abortController.signal.addEventListener("abort", () => {
          clearInterval(pollInterval);
          // Kill server-side process
          requestUrl({
            url: `${this.plugin.settings.apiUrl}/api/chat/cancel`,
            method: "POST",
            headers: apiHeaders(this.plugin, { "Content-Type": "application/json" }),
            body: JSON.stringify({ request_id: pollId }),
          }).catch(() => {});
          reject(new DOMException("Aborted", "AbortError"));
        });
      });

      loadingEl.remove();

      if (pollResult.status === "error") {
        this.addSystemMessage(`錯誤：${pollResult.error}`);
      } else {
        this.history.push({ role: "user", content: text });
        this.history.push({ role: "assistant", content: pollResult.reply });

        const u = pollResult.usage || {};
        this.totalUsage.input += (u.input_tokens || 0);
        this.totalUsage.output += (u.output_tokens || 0);
        this.totalUsage.cost += (u.cost_usd || 0);
        this.updateUsageDisplay();

        this.addAssistantMessage(pollResult.reply, pollResult.context_notes || [], u, activeNoteCtx);

        this.selectedNotes.clear();
        this.mentionedNotes = [];
        this.updateMentionChips();

        if (this.history.length > 40) {
          this.history = this.history.slice(-40);
        }
      }
    } catch (e) {
      loadingEl.remove();
      console.error("[vault-chat] sendMessage error:", e, "name:", e.name, "message:", e.message, "status:", e.status);
      if (e.name === "AbortError" || (this.abortController && this.abortController.signal.aborted)) {
        this.addSystemMessage("已停止回應");
      } else {
        const msg = e.message || String(e);
        if (msg.includes("ECONNREFUSED") || msg.includes("net::") || msg.includes("fetch") || msg.includes("Failed to fetch") || msg.includes("NetworkError")) {
          this.addSystemMessage("無法連線到伺服器 — 請確認 api_server.py 正在執行");
        } else {
          this.addSystemMessage(`錯誤：${msg}`);
        }
      }
    } finally {
      this.isWaiting = false;
      this.abortController = null;
    }
  }

  // --- Save conversation (P4) ---

  async saveConversation() {
    if (this.history.length === 0) {
      this.addSystemMessage("沒有對話可以儲存");
      return;
    }

    // Auto-generate title from first user message
    const firstUserMsg = this.history.find((m) => m.role === "user");
    const titleRaw = firstUserMsg ? firstUserMsg.content.slice(0, 30).replace(/[\\/:*?"<>|]/g, "") : "untitled";
    const title = titleRaw.trim() || "untitled";

    const now = new Date();
    const dateStr = now.toISOString().slice(0, 10);
    const filename = `${dateStr}_${title}.md`;
    const folderPath = this.plugin.settings.chatHistoryFolder || "VaultChatHistory";
    const filePath = `${folderPath}/${filename}`;

    // Ensure folder exists
    const folderExists = this.app.vault.getAbstractFileByPath(folderPath);
    if (!folderExists) {
      await this.app.vault.createFolder(folderPath);
    }

    // Build content
    const lines = [];
    lines.push("---");
    lines.push("tags:");
    lines.push("  - source/vault-chat");
    lines.push(`created: ${now.toISOString()}`);
    lines.push("---");
    lines.push("");
    lines.push(`# ${title}`);
    lines.push("");

    for (let i = 0; i < this.history.length; i++) {
      const msg = this.history[i];
      if (msg.role === "user") {
        lines.push("## User");
        lines.push(msg.content);
        lines.push("");
      } else if (msg.role === "assistant") {
        lines.push("## Assistant");
        lines.push(msg.content);
        lines.push("");
      }
      // Add separator between turns
      if (i < this.history.length - 1 && msg.role === "assistant") {
        lines.push("---");
        lines.push("");
      }
    }

    // Usage footer
    lines.push("---");
    lines.push("");
    const inK = (this.totalUsage.input / 1000).toFixed(1);
    const outK = (this.totalUsage.output / 1000).toFixed(1);
    const cost = this.totalUsage.cost.toFixed(4);
    lines.push(`Token usage: ${inK}K in / ${outK}K out · $${cost}`);

    const content = lines.join("\n");

    // Check if file already exists
    const existing = this.app.vault.getAbstractFileByPath(filePath);
    if (existing) {
      await this.app.vault.modify(existing, content);
    } else {
      await this.app.vault.create(filePath, content);
    }

    this.addSystemMessage(`對話已儲存到 ${filePath}`);
  }
}

// --- Settings Tab ---
class VaultSearchSettingTab extends PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display() {
    const { containerEl } = this;
    containerEl.empty();

    new Setting(containerEl)
      .setName("API Server URL")
      .setDesc("vault-search API server 的位址（本機: http://localhost:3789）")
      .addText((text) =>
        text
          .setPlaceholder("http://localhost:3789")
          .setValue(this.plugin.settings.apiUrl)
          .onChange(async (value) => {
            this.plugin.settings.apiUrl = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("API Key")
      .setDesc("API 伺服器認證金鑰（可留空）")
      .addText((text) => {
        text.inputEl.type = "password";
        text
          .setPlaceholder("sk-...")
          .setValue(this.plugin.settings.apiKey)
          .onChange(async (value) => {
            this.plugin.settings.apiKey = value;
            await this.plugin.saveSettings();
          });
      });

    new Setting(containerEl)
      .setName("搜尋結果數量")
      .setDesc("每次搜尋回傳的最大結果數")
      .addSlider((slider) =>
        slider
          .setLimits(5, 30, 5)
          .setValue(this.plugin.settings.nResults)
          .setDynamicTooltip()
          .onChange(async (value) => {
            this.plugin.settings.nResults = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("Chat：使用 Vault 上下文")
      .setDesc("聊天時自動搜尋相關筆記作為回答參考（RAG）")
      .addToggle((toggle) =>
        toggle
          .setValue(this.plugin.settings.useVaultContext)
          .onChange(async (value) => {
            this.plugin.settings.useVaultContext = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("預設聊天模式")
      .setDesc("Chat 面板開啟時的預設模式")
      .addDropdown((dropdown) =>
        dropdown
          .addOption("vault", "📚 Vault — notes only")
          .addOption("free", "💬 Free — no vault search")
          .addOption("hybrid", "🔀 Hybrid — auto-decide")
          .setValue(this.plugin.settings.chatMode)
          .onChange(async (value) => {
            this.plugin.settings.chatMode = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("Related Notes：結果數量")
      .setDesc("Related Notes 面板顯示的最大結果數")
      .addSlider((slider) =>
        slider
          .setLimits(5, 20, 5)
          .setValue(this.plugin.settings.contextNResults)
          .setDynamicTooltip()
          .onChange(async (value) => {
            this.plugin.settings.contextNResults = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("排除資料夾")
      .setDesc("@ 提及和搜尋時排除的資料夾（逗號分隔）")
      .addText((text) =>
        text
          .setPlaceholder(".obsidian, .trash")
          .setValue(this.plugin.settings.excludeFolders)
          .onChange(async (value) => {
            this.plugin.settings.excludeFolders = value;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("對話儲存資料夾")
      .setDesc("Chat 面板「💾 儲存對話」寫入的資料夾（相對於 vault 根目錄）")
      .addText((text) =>
        text
          .setPlaceholder("VaultChatHistory")
          .setValue(this.plugin.settings.chatHistoryFolder)
          .onChange(async (value) => {
            this.plugin.settings.chatHistoryFolder = value;
            await this.plugin.saveSettings();
          })
      );
  }
}

// --- Plugin ---
class VaultSearchPlugin extends Plugin {
  async onload() {
    await this.loadSettings();

    this.registerView(SEARCH_VIEW_TYPE, (leaf) => new VaultSearchView(leaf, this));
    this.registerView(CHAT_VIEW_TYPE, (leaf) => new VaultChatView(leaf, this));
    this.registerView(CONTEXT_VIEW_TYPE, (leaf) => new VaultContextView(leaf, this));

    this.addRibbonIcon("search", "Vault Semantic Search", () => {
      this.activateView(SEARCH_VIEW_TYPE);
    });

    this.addRibbonIcon("message-circle", "Vault Chat", () => {
      this.activateView(CHAT_VIEW_TYPE);
    });

    this.addRibbonIcon("git-compare", "Related Notes", () => {
      this.activateView(CONTEXT_VIEW_TYPE);
    });

    this.addCommand({
      id: "open-vault-search",
      name: "Open Vault Semantic Search",
      callback: () => this.activateView(SEARCH_VIEW_TYPE),
    });

    this.addCommand({
      id: "open-vault-chat",
      name: "Open Vault Chat",
      callback: () => this.activateView(CHAT_VIEW_TYPE),
    });

    this.addCommand({
      id: "open-related-notes",
      name: "Open Related Notes",
      callback: () => this.activateView(CONTEXT_VIEW_TYPE),
    });

    this.addSettingTab(new VaultSearchSettingTab(this.app, this));
  }

  async activateView(viewType) {
    const existing = this.app.workspace.getLeavesOfType(viewType);
    if (existing.length > 0) {
      this.app.workspace.revealLeaf(existing[0]);
      return;
    }
    const leaf = this.app.workspace.getRightLeaf(false);
    await leaf.setViewState({ type: viewType, active: true });
    this.app.workspace.revealLeaf(leaf);
  }

  async loadSettings() {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings() {
    await this.saveData(this.settings);
  }
}

module.exports = VaultSearchPlugin;
