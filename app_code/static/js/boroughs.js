(function () {
    const table = document.getElementById('boroughTable');
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.querySelectorAll('tr[data-name]'));
    const emptyRow = tbody.querySelector('.empty-row');
    const search = document.getElementById('boroughSearch');
    const count = document.getElementById('boroughCount');
    const filterButtons = document.querySelectorAll('.filters button');

    let tier = 'all';
    let sortKey = 'multiplier';
    let sortDir = -1;

    function render() {
        const term = search.value.trim().toLowerCase();
        const numeric = sortKey !== 'name';
        rows.sort((a, b) => {
            const x = a.dataset[sortKey];
            const y = b.dataset[sortKey];
            const diff = numeric ? parseFloat(x) - parseFloat(y) : x.localeCompare(y);
            return diff * sortDir || a.dataset.name.localeCompare(b.dataset.name);
        });
        let shown = 0;
        rows.forEach((row) => {
            const match = (tier === 'all' || row.dataset.tier === tier) && row.dataset.name.toLowerCase().includes(term);
            row.hidden = !match;
            if (match) shown++;
            tbody.insertBefore(row, emptyRow);
        });
        emptyRow.hidden = shown > 0;
        count.textContent = shown + (shown === 1 ? ' borough' : ' boroughs');
    }

    table.querySelectorAll('th[data-key] button').forEach((btn) => {
        btn.addEventListener('click', () => {
            const th = btn.parentElement;
            const key = th.dataset.key;
            if (key === sortKey) {
                sortDir = -sortDir;
            } else {
                sortKey = key;
                sortDir = key === 'name' ? 1 : -1;
            }
            table.querySelectorAll('th[aria-sort]').forEach((h) => h.removeAttribute('aria-sort'));
            th.setAttribute('aria-sort', sortDir === 1 ? 'ascending' : 'descending');
            th.querySelector('.arrow').textContent = sortDir === 1 ? '▲' : '▼';
            render();
        });
    });

    filterButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            tier = btn.dataset.tier;
            filterButtons.forEach((b) => b.setAttribute('aria-pressed', b === btn));
            render();
        });
    });

    search.addEventListener('input', render);
})();
