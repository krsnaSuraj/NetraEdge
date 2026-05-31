const path = require('path');

module.exports = {
  presets: ['module:@react-native/babel-preset'],
  plugins: [
    ['module-resolver', {
      alias: {
        '@netraedge/core': path.resolve(__dirname, '../core/src'),
        '@netraedge/react-native': path.resolve(__dirname, '../react-native/src'),
      },
    }],
  ],
};
