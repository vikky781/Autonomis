const { render, preview } = require('./common');
const slideConfig = { type: 'content', index: 7, title: 'Evaluation method' };
function createSlide(pres, theme) { return render(pres, theme, 7); }
if (require.main === module) preview(7);
module.exports = { createSlide, slideConfig };
