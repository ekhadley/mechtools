// Builds every pane of a tabbed readout in headless Chromium through the widget's script and reports each pane's structure next to the browser's parse of the Python-rendered html of the same pane, for tests/test_lens_js.py to compare.
// usage: node parity.mjs WIDGET.html PANES.json  ->  JSON on stdout: {"i,j": {built, rendered, pane, minHeight}, "initial": min-height before any switch}, a structure being
// [{header, colspan, spans, rows: [{color, cells: [{cls, text, div}]}]}] per table, with every color as the CSSOM serializes it. Exit 3 when playwright or a browser is missing (the test skips), 1 on any other error.
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
let chromium;
try { ({ chromium } = createRequire(import.meta.url)('playwright')); } catch (e) { console.error('playwright: ' + e.message); process.exit(3); }  // require, not import: NODE_PATH applies to it
const [file, panesFile] = process.argv.slice(2);
const html = readFileSync(file, 'utf8'), rendered = JSON.parse(readFileSync(panesFile, 'utf8'));
let browser;
try { browser = await chromium.launch(); } catch (e) { console.error('launch: ' + e.message); process.exit(3); }
const page = await browser.newPage();
await page.setContent(`<!doctype html><html><head><meta charset=utf-8></head><body>${html}</body></html>`);
const out = await page.evaluate((rendered) => {
  const S = root => [...root.querySelectorAll('table')].map(tb => ({
    header: tb.rows[0].cells[0].textContent, colspan: tb.rows[0].cells[0].colSpan,
    spans: [...tb.rows[0].cells[0].querySelectorAll('span')].map(s => [s.textContent, s.style.color]),
    rows: [...tb.rows].slice(1).map(tr => ({ color: tr.style.color, cells: [...tr.cells].map(td => ({ cls: td.className, text: td.textContent, div: td.firstElementChild ? td.firstElementChild.tagName : null })) })) }));
  const w = document.querySelector('[tabindex]'), bars = [...w.querySelectorAll('.tb')], box = w.querySelector('.pn');
  const out = { initial: box.style.minHeight };
  for (const k in rendered) {
    const [i, j] = k.split(',').map(Number);
    bars[0].children[i].click(); bars[1].children[j].click();
    const tmp = document.createElement('div'); tmp.innerHTML = rendered[k];
    out[k] = { built: S(box), rendered: S(tmp), pane: box.firstElementChild.offsetHeight, minHeight: box.style.minHeight };
  }
  return out;
}, rendered);
await browser.close();
console.log(JSON.stringify(out));
