const path = require('path');

module.exports = {
  project: {
    ios: {
      sourceDir: path.resolve(__dirname, 'ios'),
    },
    android: {
      sourceDir: path.resolve(__dirname, 'android'),
    },
  },
  dependencies: {
    '@netraedge/react-native': {
      root: path.resolve(__dirname, '../react-native'),
      platforms: {
        android: {
          sourceDir: path.resolve(__dirname, '../react-native/android'),
          packageImportPath: 'import com.netraedge.NetraEdgePackage',
          packageInstance: 'new NetraEdgePackage()',
        },
        ios: {
          podspecPath: path.resolve(__dirname, '../react-native/NetraEdge.podspec'),
        },
      },
    },
  },
};
