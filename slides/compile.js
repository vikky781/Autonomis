const pptxgen = require('pptxgenjs');
const fs = require('node:fs');
const path = require('node:path');

const pres = new pptxgen();
pres.defineLayout({ name: 'AUTONOMIS_WIDE', width: 10, height: 5.625 });
pres.layout = 'AUTONOMIS_WIDE';
pres.author = 'Autonomis team';
pres.subject = 'Utkrishta 2026 Challenge 1: budget-constrained churn prediction';
pres.title = 'Autonomis — Make every experiment count';
pres.company = 'Autonomis';
pres.lang = 'en-US';
pres.theme = { headFontFace: 'Aptos Display', bodyFontFace: 'Aptos', lang: 'en-US' };

const theme = {
  primary: '000814', secondary: '001D3D', accent: '003566', light: 'FFC300', bg: '000814',
};
const outDir = path.join(__dirname, 'output');
fs.mkdirSync(outDir, { recursive: true });
for (let i = 1; i <= 10; i += 1) {
  const num = String(i).padStart(2, '0');
  require(`./slide-${num}.js`).createSlide(pres, theme);
}
pres.writeFile({ fileName: path.join(outDir, 'Autonomis_Utkrishta_2026.pptx') });
