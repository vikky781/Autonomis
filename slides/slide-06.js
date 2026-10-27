const { render, preview } = require('./common');
const slideConfig = { type: 'content', index: 6, title: 'Experiment history' };
function createSlide(pres, theme) { return render(pres, theme, 6); }
if (require.main === module) preview(6);
module.exports = { createSlide, slideConfig };
