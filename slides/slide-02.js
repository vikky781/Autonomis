const { render, preview } = require('./common');
const slideConfig = { type: 'content', index: 2, title: 'The challenge' };
function createSlide(pres, theme) { return render(pres, theme, 2); }
if (require.main === module) preview(2);
module.exports = { createSlide, slideConfig };
