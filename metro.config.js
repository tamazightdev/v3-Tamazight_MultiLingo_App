// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Add TypeScript extensions to the resolver
config.resolver.sourceExts = ['jsx', 'js', 'ts', 'tsx', 'json'];

// Add explicit node modules paths for better resolution
config.resolver.nodeModulesPaths = [
  path.resolve(__dirname, 'node_modules'),
  path.resolve(__dirname, '../../node_modules'), // For pnpm workspace
];

// Keep only the main Supabase package in extraNodeModules
config.resolver.extraNodeModules = {
  '@supabase/supabase-js': path.resolve(__dirname, 'node_modules/@supabase/supabase-js'),
};

// Watch only the main node_modules directory and the main Supabase package
config.watchFolders = [
  path.resolve(__dirname, 'node_modules'),
];

// Add transformer configuration to handle TypeScript files
config.transformer = {
  ...config.transformer,
  getTransformOptions: async () => ({
    transform: {
      experimentalImportSupport: false,
      inlineRequires: true,
    },
  }),
};

module.exports = config;