/* 动手学深度学习 · 中文在线版 — 阅读增强脚本（无依赖） */
(function () {
  "use strict";

  var STORAGE_KEY = "d2l-zh-reading-position";
  var doc = document;
  var root = doc.documentElement;

  /* d2l 主题的滚动容器是 .mdl-layout__content，而不是 window */
  var scroller = doc.querySelector(".mdl-layout__content");

  function scrollTop() {
    return scroller ? scroller.scrollTop : window.scrollY;
  }

  function maxScroll() {
    if (scroller) {
      return Math.max(0, scroller.scrollHeight - scroller.clientHeight);
    }
    return Math.max(0, doc.body.scrollHeight - window.innerHeight);
  }

  function scrollToY(y) {
    if (scroller) {
      scroller.scrollTo({ top: y, behavior: "smooth" });
    } else {
      window.scrollTo({ top: y, behavior: "smooth" });
    }
  }

  function addScrollListener(fn) {
    (scroller || window).addEventListener("scroll", fn, { passive: true });
  }

  /* ---------- 阅读进度条 ---------- */
  var progress = doc.createElement("div");
  progress.id = "d2l-progress";
  doc.body.appendChild(progress);

  /* ---------- 回到顶部 ---------- */
  var topBtn = doc.createElement("button");
  topBtn.id = "d2l-top";
  topBtn.type = "button";
  topBtn.title = "回到顶部";
  topBtn.setAttribute("aria-label", "回到顶部");
  topBtn.innerHTML = "&#8593;";
  doc.body.appendChild(topBtn);
  topBtn.addEventListener("click", function () {
    scrollToY(0);
  });

  /* ---------- 继续阅读 ---------- */
  var resumeBtn = doc.createElement("button");
  resumeBtn.id = "d2l-resume";
  resumeBtn.type = "button";
  doc.body.appendChild(resumeBtn);
  var resumeTarget = null;
  var resumeTimer = null;

  function pageKey() {
    return location.pathname.replace(/index\.html$/, "");
  }

  function readSaved() {
    try {
      var raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return null;
      var data = JSON.parse(raw);
      if (!data || data.path !== pageKey()) return null;
      if (typeof data.percent !== "number") return null;
      return data;
    } catch (e) {
      return null;
    }
  }

  function savePosition() {
    var max = maxScroll();
    if (max < 600) return;
    var percent = Math.min(100, Math.max(0, (scrollTop() / max) * 100));
    try {
      localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({
          path: pageKey(),
          percent: Math.round(percent * 10) / 10,
          title: (doc.title || "").split(" \u2014 ")[0],
          ts: Date.now(),
        })
      );
    } catch (e) {
      /* localStorage 不可用时静默忽略 */
    }
  }

  function offerResume() {
    var saved = readSaved();
    if (!saved || saved.percent < 4 || saved.percent > 96) return;
    if (scrollTop() > 40) return;

    resumeTarget = saved.percent;
    resumeBtn.innerHTML =
      "继续上次阅读 · <b>" + Math.round(saved.percent) + "%</b>";
    resumeBtn.classList.add("show");
    resumeBtn.addEventListener("click", function () {
      scrollToY((saved.percent / 100) * maxScroll());
      hideResume();
    });
    resumeTimer = window.setTimeout(hideResume, 9000);
  }

  function hideResume() {
    resumeBtn.classList.remove("show");
    if (resumeTimer) {
      window.clearTimeout(resumeTimer);
      resumeTimer = null;
    }
  }

  /* ---------- 滚动事件（节流） ---------- */
  var ticking = false;
  var saveTimer = null;

  function onScroll() {
    if (ticking) return;
    ticking = true;
    window.requestAnimationFrame(function () {
      var max = maxScroll();
      var top = scrollTop();
      var ratio = max > 0 ? top / max : 0;
      progress.style.width = Math.min(100, Math.max(0, ratio * 100)) + "%";
      topBtn.classList.toggle("show", top > 700);
      if (resumeBtn.classList.contains("show") && top > 60) {
        hideResume();
      }
      ticking = false;
    });

    if (saveTimer) window.clearTimeout(saveTimer);
    saveTimer = window.setTimeout(savePosition, 600);
  }

  addScrollListener(onScroll);
  window.addEventListener("resize", onScroll, { passive: true });
  window.addEventListener("pagehide", savePosition);

  /* ---------- 界面文案中文化 ---------- */
  function localizeChrome() {
    var tocTitles = doc.querySelectorAll(
      ".mdl-layout__drawer .mdl-layout-title.toc, .side-doc-outline .caption-text"
    );
    tocTitles.forEach(function (el) {
      if (/Table Of Contents/i.test(el.textContent)) {
        el.textContent = "目录";
      }
    });

    var search = doc.querySelector('input[name="q"]');
    if (search && /^Search$/i.test(search.getAttribute("placeholder") || "")) {
      search.setAttribute("placeholder", "搜索");
    }
    var quick = doc.querySelector(".mdl-tooltip");
    if (quick && /Quick search/i.test(quick.textContent)) {
      quick.textContent = "快速搜索";
    }

    /* 上一节 / 下一节 */
    doc.querySelectorAll(".pagenation-direction").forEach(function (el) {
      var text = el.textContent.trim();
      if (text === "Previous") {
        el.textContent = "上一节";
      } else if (text === "Next") {
        el.textContent = "下一节";
      }
    });
  }

  /* ---------- 键盘导航 ---------- */
  function isTyping(target) {
    if (!target) return false;
    var tag = target.tagName;
    return (
      tag === "INPUT" ||
      tag === "TEXTAREA" ||
      tag === "SELECT" ||
      target.isContentEditable
    );
  }

  doc.addEventListener("keydown", function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey || isTyping(e.target)) return;

    if (e.key === "/") {
      var input = doc.querySelector('input[name="q"]');
      if (input) {
        e.preventDefault();
        input.focus();
      }
      return;
    }

    if (e.key === "Escape") {
      doc.querySelectorAll("span.highlighted").forEach(function (el) {
        el.classList.remove("highlighted");
      });
      try {
        localStorage.removeItem("sphinx_highlight_terms");
      } catch (err) {
        /* ignore */
      }
      return;
    }

    if (e.key === "ArrowRight") {
      var next = doc.querySelector('link[rel="next"]');
      if (next && next.href) location.href = next.href;
    } else if (e.key === "ArrowLeft") {
      var prev = doc.querySelector('link[rel="prev"]');
      if (prev && prev.href) location.href = prev.href;
    }
  });

  /* ---------- 中文搜索分词 ----------
   * Sphinx 的浏览器端分词只按空白切分，而索引是按 jieba 词切分的，
   * 因此「线性回归」这类查询会查不到。这里用索引词表做正向最大匹配，
   * 把查询切成与索引一致的词。
   */
  function installChineseSplitter() {
    var vocab = window.D2L_SEARCH_VOCAB;
    if (!Array.isArray(vocab) || !vocab.length) return;

    var words = Object.create(null);
    var maxLen = 2;
    vocab.forEach(function (w) {
      words[w] = true;
      if (w.length > maxLen) maxLen = w.length;
    });

    var cjk = /[\u3400-\u9fff]/;

    function segment(run) {
      var out = [];
      var i = 0;
      var produced = false;
      while (i < run.length) {
        var matched = null;
        for (var len = Math.min(maxLen, run.length - i); len >= 2; len--) {
          var cand = run.substr(i, len);
          if (words[cand]) {
            matched = cand;
            break;
          }
        }
        if (matched) {
          out.push(matched);
          i += matched.length;
          produced = true;
        } else {
          i += 1;
        }
      }
      return produced ? out : [run];
    }

    window.splitQuery = function (query) {
      var tokens = String(query || "")
        .split(/[^\p{Letter}\p{Number}_\p{Emoji_Presentation}]+/gu)
        .filter(Boolean);
      var result = [];
      tokens.forEach(function (token) {
        if (token.length <= 2 || !cjk.test(token)) {
          result.push(token);
          return;
        }
        result = result.concat(segment(token));
      });
      return result;
    };
  }

  /* ---------- 启动 ---------- */
  installChineseSplitter();
  localizeChrome();
  offerResume();
  onScroll();
})();
