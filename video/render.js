// Deterministic frame renderer: seeks the GSAP timeline in src/index.html and screenshots each frame.
//   [SRC=long.html] node render.js still <t1> <t2> ...      -> stills/<t>.png  (for review)
//   [SRC=long.html] node render.js frames [fps=30] [workers=4] -> frames/%04d.png   (duration read from the page)
const path = require('path');
const fs = require('fs');
const { chromium } = require('playwright');

const SRC = 'file://' + path.resolve(__dirname, 'src', process.env.SRC || 'index.html');
const [mode, ...args] = process.argv.slice(2);

async function open(browser) {
  const page = await browser.newPage({ viewport: { width: 1920, height: 1080 }, deviceScaleFactor: 1 });
  await page.goto(SRC);
  await page.waitForFunction('window.__ready === true', null, { timeout: 30000 });
  await page.evaluate(() => Promise.all([
    document.fonts.load('800 100px "Plus Jakarta Sans"'), document.fonts.load('600 30px "Plus Jakarta Sans"'),
    document.fonts.load('700 30px "Inter"'), document.fonts.load('500 30px "Inter"'),
  ]));
  return page;
}

(async () => {
  const launchArgs = { args: ['--font-render-hinting=none', '--disable-lcd-text', '--force-color-profile=srgb'] };
  if (mode === 'still') {
    fs.mkdirSync(path.join(__dirname, 'stills'), { recursive: true });
    const browser = await chromium.launch(launchArgs);
    const page = await open(browser);
    for (const t of args.map(Number)) {
      await page.evaluate(t => window.seek(t), t);
      await page.screenshot({ path: path.join(__dirname, 'stills', t.toFixed(2) + '.png') });
    }
    await browser.close();
  } else if (mode === 'frames') {
    const fps = Number(args[0] || 30), workers = Number(args[1] || 4);
    const dur = await (async () => { const b = await chromium.launch(launchArgs); const pg = await open(b); const d = await pg.evaluate(() => window.__DURATION); await b.close(); return d; })();
    const total = Math.round(dur * fps);
    const dir = path.join(__dirname, 'frames');
    fs.rmSync(dir, { recursive: true, force: true }); fs.mkdirSync(dir, { recursive: true });
    const per = Math.ceil(total / workers);
    await Promise.all(Array.from({ length: workers }, async (_, w) => {
      const browser = await chromium.launch(launchArgs);
      const page = await open(browser);
      for (let i = w * per; i < Math.min(total, (w + 1) * per); i++) {
        await page.evaluate(t => window.seek(t), i / fps);
        await page.screenshot({ path: path.join(dir, String(i).padStart(4, '0') + '.png') });
      }
      await browser.close();
    }));
    console.log('rendered', total, 'frames @', fps, 'fps');
  }
})().catch(e => { console.error(e); process.exit(1); });
