const { render, preview } = require('./common');
const slideConfig = { type: 'content', index: 8, title: 'Results' };
function createSlide(pres, theme) { return render(pres, theme, 8); }
if (require.main === module) preview(8);
module.exports = { createSlide, slideConfig };
