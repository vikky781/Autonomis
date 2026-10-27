const { render, preview } = require('./common');
const slideConfig = { type: 'content', index: 3, title: 'Data and preprocessing' };
function createSlide(pres, theme) { return render(pres, theme, 3); }
if (require.main === module) preview(3);
module.exports = { createSlide, slideConfig };
