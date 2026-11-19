//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

function escapeHtml(text) {
    return String(text).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}

function promptHtml(text) {
    return escapeHtml(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
}

const NO_VIDEOS = `<div class="loading-message">No videos available.</div>`;

function categoryTabs(categories, onPick) {
    const nav = document.createElement("div");
    nav.className = "dataset-nav";
    nav.innerHTML = `<ul class="nav nav-tabs" role="tablist">${categories.map((c, i) =>
        `<li class="nav-item" role="presentation">
             <button class="nav-link${i ? "" : " active"}" type="button" role="tab" aria-selected="${!i}" data-index="${i}">${escapeHtml(c.name)}</button>
         </li>`).join("")}</ul>`;
    const buttons = [...nav.querySelectorAll(".nav-link")];
    buttons.forEach((el, i) => el.addEventListener("click", () => {
        buttons.forEach((b, j) => { b.classList.toggle("active", j === i); b.setAttribute("aria-selected", String(j === i)); });
        onPick(i);
    }));
    if (categories.length === 1) nav.hidden = true;
    return nav;
}

function playVideo(root, category, index, autoplay = true) {
    const video = category.videos[index];
    root.querySelector(".player-area").innerHTML =
        `<video class="fade-in" src="${encodeURI(video.file)}" ${video.thumb ? `poster="${encodeURI(video.thumb)}"` : ""}
                controls ${autoplay ? "autoplay" : ""} muted loop playsinline preload="none"></video>
         <p class="gallery-prompt">${video.prompt ? `“${promptHtml(video.prompt)}”` : escapeHtml(video.title)}</p>`;
    root.querySelectorAll(".thumbnail-item").forEach((el, i) => {
        el.classList.toggle("active", i === index);
    });
}

function showCategory(root, category, autoplay = true) {
    const grid = root.querySelector(".thumbnail-row");
    grid.innerHTML = category.videos.map((video, i) =>
        `<div class="thumbnail-item" title="${escapeHtml(video.title)}" data-index="${i}">
             ${video.thumb ? `<img src="${encodeURI(video.thumb)}" alt="${escapeHtml(video.title)}" loading="lazy" decoding="async">` : ""}
             <div class="thumbnail-title">${escapeHtml(video.title)}</div>
         </div>`).join("");
    grid.querySelectorAll(".thumbnail-item").forEach(el => {
        el.addEventListener("click", () => playVideo(root, category, Number(el.dataset.index)));
    });
    if (category.videos.length) playVideo(root, category, 0, autoplay);
}

function mountPlayer(root, categories) {
    root.innerHTML = `<div class="thumbnail-container"><div class="thumbnail-row"></div></div><div class="player-area"></div>`;
    root.prepend(categoryTabs(categories, i => showCategory(root, categories[i])));
    showCategory(root, categories[0], false);
}

const REDUCED_MOTION = window.matchMedia("(prefers-reduced-motion: reduce)");

const ZOOM_COLUMNS = 2.5;

function galleryGrid(category) {
    return `<ul class="gal">${category.videos.map(video => {
        const zoomed = video.prompt ? `<span class="gal-line">${promptHtml(video.prompt)}</span>` : "";
        const hold = END_HOLD[video.benchmark];
        return `<li tabindex="0">
             <figure>
                 <video class="gal-vid" data-src="${encodeURI(video.card || video.file)}" ${video.card ? `data-full="${encodeURI(video.file)}"` : ""} ${video.thumb ? `data-poster="${encodeURI(video.thumb)}"` : ""}
                        muted ${hold ? `data-hold="${hold}"` : "loop"} playsinline preload="none" aria-label="${escapeHtml(video.title)}"></video>
                 <figcaption><b>${escapeHtml(video.short_title || video.title)}</b></figcaption>
             </figure>
             ${zoomed ? `<p class="gal-prompt">${zoomed}</p>` : ""}
             ${video.benchmark ? `<span class="gal-bench">${escapeHtml(video.benchmark)}</span>` : ""}
         </li>`;
    }).join("")}</ul>`;
}

function setZoomOrigins(grid) {
    const items = [...grid.children];
    if (!items.length || !grid.offsetParent) return;
    const cols = getComputedStyle(grid).gridTemplateColumns.split(" ").length;
    const w = items[0].offsetWidth, h = items[0].offsetHeight, gap = parseFloat(getComputedStyle(grid).columnGap) || 0;
    const span = Math.min(parseFloat(grid.closest(".amp-gallery")?.dataset.zoomColumns) || ZOOM_COLUMNS, cols);
    const s = (span * w + (Math.ceil(span) - 1) * gap) / w;
    grid.style.setProperty("--zoom", s.toFixed(3));
    if (s <= 1) return;
    const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
    const origin = (x, size, total) => {
        const target = size * s > total ? 0 : clamp(x - (s - 1) * size / 2, 0, total - size * s);
        return (x - target) / (s - 1);
    };
    const px = v => `${v.toFixed(1)}px`;
    items.forEach(li => {
        const ox = origin(li.offsetLeft, w, grid.clientWidth), oy = origin(li.offsetTop, h, grid.clientHeight);
        li.style.setProperty("--ox", px(ox));
        li.style.setProperty("--oy", px(oy));
        li.style.setProperty("--zx", px(ox * (1 - s)));
        li.style.setProperty("--zt", px(oy * (1 - s)));
        li.style.setProperty("--zb", px(oy * (1 - s) + s * h - h));
        li.style.setProperty("--zw", px(s * w));
    });
}

function keepZoomOrigins(grid) {
    let pending = 0;
    const update = () => { if (!pending) pending = requestAnimationFrame(() => { pending = 0; setZoomOrigins(grid); }); };
    if ("ResizeObserver" in window) new ResizeObserver(update).observe(grid);
    else window.addEventListener("resize", update);
    setZoomOrigins(grid);
}

const wanted = new WeakMap();
const SLOTS = 3, AHEAD = 4;
const band = new Set(), ahead = new Set(), loading = new Set();
const distance = v => { const r = v.getBoundingClientRect(); return Math.abs(r.top + r.height / 2 - innerHeight / 2); };
const filled = v => v.buffered.length > 0 && v.duration > 0 && v.buffered.end(v.buffered.length - 1) >= v.duration - 0.25;
const play = v => { const p = v.play(); if (p) p.catch(() => {}); };
const attach = v => { if (!v.getAttribute("src")) v.src = v.dataset.src; };

function release(video) {
    if (loading.delete(video)) pump();
}

function begin(video) {
    attach(video);
    video.preload = "auto";
    video.dataset.go = "1";
    loading.add(video);
    const done = () => release(video);
    video.addEventListener("canplaythrough", done, {once: true});
    video.addEventListener("error", done, {once: true});
    setTimeout(done, 6000);
}

function pump() {
    const open = () => SLOTS - loading.size;
    for (const v of [...band].filter(v => wanted.get(v)).sort((x, y) => distance(x) - distance(y))) {
        if (!v.dataset.go) {
            if (open() <= 0) { v.parentElement?.classList.add("is-loading"); continue; }
            begin(v);
        }
        if (v.paused) play(v);
    }
    const warm = [...ahead].filter(v => !band.has(v) && !v.dataset.go).sort((x, y) => distance(x) - distance(y)).slice(0, AHEAD);
    for (const v of warm) if (open() > 0) begin(v);
}

const startClip = video => { attach(video); play(video); };
const onScreen = ("IntersectionObserver" in window) ? new IntersectionObserver(entries => entries.forEach(e => {
    const v = e.target;
    if (e.isIntersecting) { band.add(v); pump(); return; }
    band.delete(v);
    v.pause();
    v.parentElement?.classList.remove("is-loading");
    release(v);
    if (v.dataset.go && !filled(v)) { v.removeAttribute("src"); v.removeAttribute("data-go"); v.load(); }
}), {rootMargin: "200px 0px"}) : null;

const showPoster = video => { if (video.dataset.poster) video.poster = video.dataset.poster; };
const nearScreen = ("IntersectionObserver" in window) ? new IntersectionObserver((entries, observer) => entries.forEach(e => {
    if (e.isIntersecting) { showPoster(e.target); if (REDUCED_MOTION.matches) attach(e.target); observer.unobserve(e.target); }
}), {rootMargin: "100% 0px"}) : null;

function prefetchClips() {
    if (!("IntersectionObserver" in window)) return;
    const watch = new IntersectionObserver(entries => { entries.forEach(e => e.isIntersecting ? ahead.add(e.target) : ahead.delete(e.target)); pump(); }, {rootMargin: "150% 0px"});
    document.querySelectorAll(".gal-vid").forEach(video => watch.observe(video));
}
window.addEventListener("scroll", prefetchClips, {once: true, passive: true});

function wireVideo(video) {
    wanted.set(video, !REDUCED_MOTION.matches);
    if (REDUCED_MOTION.matches) video.controls = true;
    if (video.dataset.hold) holdLastFrame(video, Number(video.dataset.hold));
    const toggle = () => {
        if (video.controls) return;
        const play = video.paused && !(video.ended && wanted.get(video));
        wanted.set(video, play);
        play ? startClip(video) : video.pause();
    };
    video.closest("li").addEventListener("keydown", e => {
        if (e.target === e.currentTarget && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); toggle(); }
    });
    if (nearScreen) nearScreen.observe(video);
    else showPoster(video);
    if (onScreen) onScreen.observe(video);
    else if (wanted.get(video)) startClip(video);
}

function sharpen(li, on) {
    const small = li.querySelector(".gal-vid");
    let big = li.querySelector(".gal-big");
    if (!on) { big?.remove(); return; }
    if (big || !small?.dataset.full || small.paused) return;
    big = document.createElement("video");
    big.className = "gal-big";
    big.muted = true;
    big.playsInline = true;
    big.loop = !small.dataset.hold;
    big.src = small.dataset.full;
    big.addEventListener("loadeddata", () => { big.currentTime = small.currentTime; play(big); }, {once: true});
    big.addEventListener("playing", () => big.classList.add("live"), {once: true});
    small.after(big);
}

function trackZoom(grid) {
    let hot = null, timer = 0;
    const zoom = li => {
        if (li === hot) return;
        clearTimeout(timer);
        if (hot) sharpen(hot, false);
        hot?.classList.remove("zoom");
        hot = li;
        li?.classList.add("zoom");
        if (li) timer = setTimeout(() => sharpen(li, true), 200);
    };
    const inside = (r, x, y, pad = 0) => x >= r.left - pad && x < r.right + pad && y >= r.top - pad && y < r.bottom + pad;
    grid.addEventListener("pointermove", e => {
        if (e.pointerType !== "mouse") return;
        const pad = (parseFloat(getComputedStyle(grid).columnGap) || 0) / 2;
        const cell = [...grid.children].find(li => inside(li.getBoundingClientRect(), e.clientX, e.clientY, pad));
        if (cell) zoom(cell);
        else if (!hot || !inside(hot.querySelector("figure").getBoundingClientRect(), e.clientX, e.clientY)) zoom(null);
    });
    grid.addEventListener("pointerleave", () => zoom(null));
}

function mountGrid(root, categories) {
    if (root.dataset.columns) root.style.setProperty("--gal-cols", root.dataset.columns);
    root.innerHTML = categories.map((c, i) => `<div class="gal-pane"${i ? " hidden" : ""}>${galleryGrid(c)}</div>`).join("");
    const panes = [...root.querySelectorAll(".gal-pane")];
    root.prepend(categoryTabs(categories, i => panes.forEach((p, j) => { p.hidden = j !== i; })));
    root.querySelectorAll(".gal").forEach(grid => { keepZoomOrigins(grid); trackZoom(grid); });
    root.querySelectorAll(".gal-vid").forEach(wireVideo);
}

const LAYOUTS = {player: mountPlayer, grid: mountGrid};

whenParsed(() => {
    const data = (typeof AMP_VIDEOS !== "undefined") ? AMP_VIDEOS : {categories: []};
    document.querySelectorAll(".amp-gallery").forEach(root => {
        const name = root.dataset.gallery;
        const categories = ((name ? data.galleries?.[name] : data)?.categories || []).filter(c => c.videos.length);
        if (!categories.length) { root.innerHTML = NO_VIDEOS; return; }
        (LAYOUTS[root.dataset.layout] || mountPlayer)(root, categories);
    });
});

document.addEventListener("DOMContentLoaded", () => {
    const links = [...document.querySelectorAll(".navbar-nav .nav-link")]
        .map(link => ({link, section: document.querySelector(link.getAttribute("href"))}))
        .filter(x => x.section);
    if (!links.length) return;

    const LINE = 80;
    let shown = null, raf = 0;

    const visible = section => {
        const r = section.getBoundingClientRect();
        return Math.min(r.bottom, window.innerHeight) - Math.max(r.top, 0);
    };

    const pick = () => {
        const last = links[links.length - 1].section;
        if (last.getBoundingClientRect().bottom <= window.innerHeight + 4) {
            let best = null, area = -1;
            for (const {link, section} of links) {
                const v = visible(section);
                if (v > area) { area = v; best = link; }
            }
            return best;
        }
        let found = null;
        for (const {link, section} of links) {
            if (section.getBoundingClientRect().top <= LINE) found = link;
        }
        return found;
    };

    const update = () => {
        raf = 0;
        const next = pick();
        if (next === shown) return;
        if (shown) shown.classList.remove("active");
        if (next) next.classList.add("active");
        shown = next;
    };
    const tick = () => { if (!raf) raf = requestAnimationFrame(update); };

    window.addEventListener("scroll", tick, {passive: true});
    window.addEventListener("resize", tick, {passive: true});
    window.addEventListener("load", tick);
    if (window.ResizeObserver) {
        const ro = new ResizeObserver(tick);
        ro.observe(document.documentElement);
        for (const {section} of links) ro.observe(section);
    }
    update();
});

// One player: the demo video starts pinned behind the title, folds into its slot over scrollY 0 -> T and is released
// there. T puts the "Demo video" heading just below the navbar. Reduced motion: no fly, no autoplay, native controls.
document.addEventListener("DOMContentLoaded", () => {
    const root = document.documentElement;
    const hero = document.querySelector(".hero");
    const vid = document.querySelector("#demo-video .demo-vid");
    if (!hero || !vid) return;
    const slot = vid.parentElement;
    const section = document.getElementById("demo-video");
    const head = section.querySelector("h2");
    const nav = document.querySelector(".navbar.fixed-top");
    const menu = nav && nav.querySelector(".navbar-collapse");
    const toggler = nav && nav.querySelector(".navbar-toggler");
    const runway = document.querySelector(".hero-runway");
    const sheet = hero.querySelector(".hero-sheet");
    const text = hero.querySelector(".hero-text");
    const cue = hero.querySelector(".hero-cue");

    const clamp = x => (x < 0 ? 0 : x > 1 ? 1 : x);
    const lerp = (a, b, t) => a + (b - a) * t;
    const ease = t => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

    let on = false, T = 1, R = 0, raf = 0, flying = false, inView = true, waiting = false, held = false, ours = false;

    const docTop = el => el.getBoundingClientRect().top + window.scrollY;
    // The bar alone: an open phone menu (resize fires as a mobile toolbar hides) must not shorten the fold.
    const barHeight = () => {
        if (!nav) return 0;
        const open = menu && toggler && getComputedStyle(toggler).display !== "none"
            && (menu.classList.contains("show") || menu.classList.contains("collapsing"));
        return nav.getBoundingClientRect().height - (open ? menu.getBoundingClientRect().height : 0);
    };
    const measure = () => {
        T = Math.max(1, docTop(head || slot) - barHeight() - 20);
        R = parseFloat(getComputedStyle(slot).borderTopLeftRadius) || 0;
    };
    const fly = yes => {
        if (yes === flying) return;
        flying = yes;
        vid.classList.toggle("is-flying", yes);
        slot.classList.toggle("is-away", yes);
        vid.controls = !yes;
        if (!yes) for (const k of ["left", "top", "width", "height", "borderRadius"]) vid.style[k] = "";
    };
    const frame = () => {
        raf = 0;
        if (!on) return;
        const p = clamp(window.scrollY / T), q = clamp(p / 0.45), e = ease(clamp((p - 0.45) / 0.55));
        const opacity = 1 - clamp(q * 1.6);
        if (text) {
            text.style.opacity = String(opacity);
            text.style.transform = `translate3d(0,${(-q * 14).toFixed(2)}svh,0)`;
            text.style.visibility = opacity < 0.02 ? "hidden" : "";
        }
        if (sheet) sheet.style.opacity = String(1 - q);
        if (cue) cue.style.opacity = String(1 - clamp(q * 3));
        if (p >= 1) { fly(false); return; }
        fly(true);
        const r = slot.getBoundingClientRect(), vw = root.clientWidth, vh = window.innerHeight;
        Object.assign(vid.style, {
            left: `${lerp(0, r.left, e)}px`, top: `${lerp(0, r.top, e)}px`,
            width: `${lerp(vw, r.width, e)}px`, height: `${lerp(vh, r.height, e)}px`,
            borderRadius: `${lerp(0, R, e)}px`,
        });
    };
    const tick = () => { if (!raf) raf = requestAnimationFrame(frame); };

    const GESTURES = ["touchend", "click", "keydown"];
    const onGesture = () => {
        stopWaiting();
        if (on && inView) start();
    };
    const stopWaiting = () => {
        waiting = false;
        GESTURES.forEach(type => window.removeEventListener(type, onGesture, true));
    };
    const waitForGesture = () => {
        if (waiting || !on) return;
        waiting = true;
        GESTURES.forEach(type => window.addEventListener(type, onGesture, {capture: true, passive: true}));
    };
    const start = () => {
        if (held) return;
        vid.preload = "auto";
        const p = vid.play();
        if (p) p.catch(e => { if (e && e.name === "NotAllowedError") waitForGesture(); });
    };
    // A pause the visitor made with the controls holds until they play again; only ours resume by themselves.
    const pause = () => { if (!vid.paused) { ours = true; vid.pause(); } };
    vid.addEventListener("pause", () => { if (!ours && !flying) held = true; ours = false; });
    vid.addEventListener("play", () => { held = false; });

    const setOn = v => {
        on = v;
        root.classList.toggle("fly-on", v);
        vid.controls = !(v && flying);
        if (v) {
            measure();
            frame();
            if (inView) start();
            return;
        }
        fly(false);
        stopWaiting();
        pause();
        for (const el of [sheet, text, cue]) if (el) el.removeAttribute("style");
    };

    // Watch the hero, the runway and the section together: while flying the video is fixed, so its own box says nothing.
    if ("IntersectionObserver" in window) {
        const seen = new Set();
        const io = new IntersectionObserver(entries => {
            for (const e of entries) e.isIntersecting ? seen.add(e.target) : seen.delete(e.target);
            const was = inView;
            inView = seen.size > 0;
            if (!inView) pause();
            else if (!was && on) start();
        });
        for (const el of [hero, runway, section]) if (el) io.observe(el);
    }

    const remeasure = () => { if (on) { measure(); tick(); } };
    window.addEventListener("scroll", tick, {passive: true});
    window.addEventListener("resize", remeasure, {passive: true});
    window.addEventListener("orientationchange", remeasure);
    window.addEventListener("load", remeasure);
    if (REDUCED_MOTION.addEventListener) REDUCED_MOTION.addEventListener("change", e => setOn(!e.matches));
    setOn(!REDUCED_MOTION.matches);

    // The runway exists only from here on, so a deep link the browser already followed lands 35vh short.
    // A reload or back/forward keeps the position the browser restored.
    const fresh = (performance.getEntriesByType?.("navigation")[0]?.type ?? "navigate") === "navigate";
    let target = null;
    try { target = on && fresh && location.hash.length > 1 && document.getElementById(decodeURIComponent(location.hash.slice(1))); } catch (e) {}
    if (target) {
        let moved = false;
        const stop = () => { moved = true; };
        ["wheel", "touchstart", "keydown", "mousedown"].forEach(type => window.addEventListener(type, stop, {once: true, passive: true}));
        const land = () => { if (!moved) target.scrollIntoView({block: "start", behavior: "instant"}); };
        land();
        window.addEventListener("load", land, {once: true});
    }
});
