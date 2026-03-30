// Re-export everything for test convenience
module.exports = {
  ...require('./core'),
  ...require('./model'),
  ...require('./optimizer'),
  ...require('./ui'),
};
