const { render, preview } = require('./common');
const slideConfig = { type: 'content', index: 4, title: 'System design' };
function createSlide(pres, theme) { return render(pres, theme, 4); }
if (require.main === module) preview(4);
module.exports = { createSlide, slideConfig };
