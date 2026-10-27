(() => {
  const groups = [...document.querySelectorAll('.sample-group')];
  const cards = [...document.querySelectorAll('.sample-card')];
  const search = document.getElementById('sample-search');
  const collection = document.getElementById('sample-collection');
  const split = document.getElementById('sample-split');
  const tested = document.getElementById('sample-tested');
  const counter = document.getElementById('sample-count');
  function filter() {
    const q = search.value.trim().toLowerCase();
    const active = Boolean(q || collection.value !== 'all' || split.value !== 'all' || tested.checked);
    let visible = 0;
    for (const card of cards) {
      const ok = (!q || card.dataset.search.includes(q)) &&
        (collection.value === 'all' || card.dataset.collection === collection.value) &&
        (split.value === 'all' || card.dataset.split === split.value) &&
        (!tested.checked || card.dataset.tested === '1');
      card.hidden = !ok;
      visible += Number(ok);
    }
    for (const group of groups) {
      const count = group.querySelectorAll('.sample-card:not([hidden])').length;
      group.hidden = count === 0;
      group.querySelector('.group-count').textContent = count;
      if (active && count) group.open = true;
    }
    counter.textContent = `当前显示 ${visible.toLocaleString()} / ${cards.length.toLocaleString()} 张/页；每个展开类型内包含全部匹配样本。`;
    document.getElementById('no-match').hidden = visible !== 0;
  }
  search.addEventListener('input', filter);
  collection.addEventListener('change', filter);
  split.addEventListener('change', filter);
  tested.addEventListener('change', filter);
  document.getElementById('expand-all').addEventListener('click', () => groups.forEach(g => {if (!g.hidden) g.open = true;}));
  document.getElementById('collapse-all').addEventListener('click', () => groups.forEach(g => g.open = false));
  document.getElementById('reset-filters').addEventListener('click', () => {
    search.value = ''; collection.value = 'all'; split.value = 'all'; tested.checked = false; filter();
  });
  function showAnchor(hash) {
    if (!hash.startsWith('#group-')) return;
    const target = document.getElementById(decodeURIComponent(hash.slice(1)));
    if (!target) return;
    if (target.hidden) {
      search.value = ''; collection.value = 'all'; split.value = 'all'; tested.checked = false; filter();
    }
    target.open = true;
  }
  document.addEventListener('click', event => {
    const anchor = event.target.closest('a[href^="#group-"]');
    if (anchor) showAnchor(anchor.getAttribute('href'));
  });
  window.addEventListener('hashchange', () => showAnchor(location.hash));
  showAnchor(location.hash);
  const viewer = document.getElementById('image-viewer');
  const canvas = viewer.querySelector('.viewer-canvas');
  const original = document.getElementById('viewer-original');
  const preview = document.getElementById('viewer-image');
  const zoom = document.getElementById('viewer-zoom');
  document.addEventListener('click', event => {
    const link = event.target.closest('.image-link');
    if (!link || event.ctrlKey || event.metaKey || event.shiftKey) return;
    event.preventDefault();
    preview.src = link.href;
    preview.alt = link.dataset.title;
    original.href = link.href;
    document.getElementById('viewer-title').textContent = link.dataset.title;
    canvas.classList.remove('zoom'); zoom.textContent = '1:1 放大';
    viewer.showModal();
  });
  zoom.addEventListener('click', () => {
    const active = canvas.classList.toggle('zoom');
    zoom.textContent = active ? '适应窗口' : '1:1 放大';
  });
  document.getElementById('viewer-close').addEventListener('click', () => viewer.close());
  viewer.addEventListener('click', event => {if (event.target === viewer) viewer.close();});
})();
