const { render, preview } = require('./common');
const slideConfig = { type: 'content', index: 5, title: 'Core algorithm' };
function createSlide(pres, theme) { return render(pres, theme, 5); }
if (require.main === module) preview(5);
module.exports = { createSlide, slideConfig };
