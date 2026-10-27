const { render, preview } = require('./common');
const slideConfig = { type: 'summary', index: 10, title: 'Limits and next steps' };
function createSlide(pres, theme) { return render(pres, theme, 10); }
if (require.main === module) preview(10);
module.exports = { createSlide, slideConfig };
