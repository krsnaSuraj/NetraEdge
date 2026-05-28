const { getDefaultConfig, mergeConfig } = require('@react-native/metro-config');
const path = require('path');

const rootDir = path.resolve(__dirname, '../..');
const packagesDir = path.resolve(rootDir, 'packages');

const config = {
  watchFolders: [rootDir],
  resolver: {
    extraNodeModules: new Proxy(
      {},
      {
        get: (target, name) => {
          return path.join(__dirname, `node_modules/${name}`);
        },
      },
    ),
    nodeModulesPaths: [
      path.resolve(__dirname, 'node_modules'),
      path.resolve(rootDir, 'node_modules'),
    ],
  },
  transformer: {
    getTransformOptions: async () => ({
      transform: {
        experimentalImportSupport: false,
        inlineRequires: true,
      },
    }),
  },
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
