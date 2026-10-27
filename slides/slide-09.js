const { render, preview } = require('./common');
const slideConfig = { type: 'content', index: 9, title: 'Product walkthrough' };
function createSlide(pres, theme) { return render(pres, theme, 9); }
if (require.main === module) preview(9);
module.exports = { createSlide, slideConfig };
