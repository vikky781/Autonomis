const { render, preview } = require('./common');
const slideConfig = { type: 'cover', index: 1, title: 'Autonomis' };
function createSlide(pres, theme) { return render(pres, theme, 1); }
if (require.main === module) preview(1);
module.exports = { createSlide, slideConfig };
