(function () {
  const cfg = window.MANUAL_COMMON;
  if (!cfg) {
    return;
  }

  const pageKey = document.body.getAttribute("data-manual-page") || "home";
  const currentGroup = document.body.getAttribute("data-manual-group") || "qa-home";
  const currentHash = window.location.hash || "";
  const currentPage = (window.location.pathname.split("/").pop() || "index.html").toLowerCase();

  const headerHost = document.querySelector("[data-manual-header]");
  const sidebarHost = document.querySelector("[data-manual-sidebar]");
  const footerHost = document.querySelector("[data-manual-footer]");

  if (headerHost) {
    const topLinks = cfg.topLinks
      .map(function (item) {
        const active = item.key === pageKey ? " class=\"active\"" : "";
        return "<a href=\"" + item.href + "\"" + active + ">" + item.label + "</a>";
      })
      .join("");

    headerHost.innerHTML = ""
      + "<header>"
      + "  <div class=\"header-row\">"
      + "    <div class=\"brand\">" + cfg.brand + "</div>"
      + "    <button class=\"menu-toggle\" data-menu-toggle type=\"button\">" + cfg.menuLabel + "</button>"
      + "    <nav class=\"top-links\" aria-label=\"Top navigation\">" + topLinks + "</nav>"
      + "  </div>"
      + "</header>";
  }

  if (sidebarHost) {
    const groups = cfg.quickAccessGroups
      .map(function (group) {
        const isOpen = group.id === currentGroup ? " open" : "";
        const items = group.items
          .map(function (item) {
            let active = "";
            if (item.key === pageKey) {
              const hashIndex = item.href.indexOf("#");
              const itemPage = (hashIndex >= 0 ? item.href.slice(0, hashIndex) : item.href).toLowerCase();
              const itemHash = hashIndex >= 0 ? item.href.slice(hashIndex) : "";
              const pageMatches = itemPage === currentPage;
              const hashMatches = (itemHash && itemHash === currentHash) || (!itemHash && !currentHash);
              if (pageMatches && hashMatches) {
                active = " class=\"active\"";
              }
            }
            return "<li><a href=\"" + item.href + "\"" + active + ">" + item.label + "</a></li>";
          })
          .join("");

        return ""
          + "<details class=\"sidebar-group\" id=\"" + group.id + "\"" + isOpen + ">"
          + "  <summary>" + group.label + "</summary>"
          + "  <ul>" + items + "</ul>"
          + "</details>";
      })
      .join("");

    sidebarHost.innerHTML = ""
      + "<div class=\"sidebar-title\">" + cfg.quickAccessTitle + "</div>"
      + groups;
  }

  if (footerHost) {
    footerHost.innerHTML = cfg.footerByPage[pageKey] || "";
  }
})();
