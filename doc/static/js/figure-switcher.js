document.querySelectorAll('[data-figure-viewer]').forEach((viewer) => {
  const tabs = Array.from(viewer.querySelectorAll('[role="tab"]'));
  const panels = Array.from(viewer.querySelectorAll('[role="tabpanel"]'));

  const selectPanel = (tab) => {
    const panelId = tab.dataset.panel;
    tabs.forEach((item) => {
      item.setAttribute('aria-selected', String(item === tab));
      item.tabIndex = item === tab ? 0 : -1;
    });
    panels.forEach((panel) => {
      panel.hidden = panel.id !== panelId;
    });
  };

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => selectPanel(tab));
    tab.addEventListener('keydown', (event) => {
      if (!['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault();
      let nextIndex = index;
      if (event.key === 'ArrowLeft') nextIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'ArrowRight') nextIndex = (index + 1) % tabs.length;
      if (event.key === 'Home') nextIndex = 0;
      if (event.key === 'End') nextIndex = tabs.length - 1;
      tabs[nextIndex].focus();
      selectPanel(tabs[nextIndex]);
    });
  });

  selectPanel(tabs.find((tab) => tab.getAttribute('aria-selected') === 'true') || tabs[0]);
});
