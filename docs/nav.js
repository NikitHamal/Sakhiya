/* Section router. Standalone so tabs work even while models load or if another script fails. */
(function () {
  var DEF = "studio";
  function go() {
    var h = (location.hash || "#" + DEF).slice(1), secs = document.querySelectorAll("main > section"), ok = false, i;
    for (i = 0; i < secs.length; i++) if (secs[i].id === h) ok = true;
    if (!ok) h = DEF;
    for (i = 0; i < secs.length; i++) secs[i].classList.toggle("on", secs[i].id === h);
    var as = document.querySelectorAll(".tabs a");
    for (i = 0; i < as.length; i++) { var on = as[i].getAttribute("href") === "#" + h; as[i].classList.toggle("on", on); if (on) as[i].setAttribute("aria-current", "page"); else as[i].removeAttribute("aria-current"); }
    try { document.dispatchEvent(new CustomEvent("sknav", { detail: h })); } catch (e) {}
  }
  document.addEventListener("click", function (e) {
    var a = e.target && e.target.closest ? e.target.closest("a[href^='#']") : null;
    if (!a) return; var h = a.getAttribute("href");
    if (h && h.length > 1) { e.preventDefault(); if (location.hash !== h) history.pushState(null, "", h); go(); window.scrollTo(0, 0); }
  });
  addEventListener("hashchange", go); addEventListener("popstate", go);
  window.skGo = go;
  addEventListener("scroll", function () { var b = document.getElementById("appbar"); if (b) b.classList.toggle("scrolled", window.scrollY > 6); }, { passive: true });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", go); else go();
})();
