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

// Add explicit module resolution for Supabase dependencies
config.resolver.extraNodeModules = {
  '@supabase/postgrest-js': path.resolve(__dirname, 'node_modules/@supabase/postgrest-js'),
  '@supabase/storage-js': path.resolve(__dirname, 'node_modules/@supabase/storage-js'),
  '@supabase/realtime-js': path.resolve(__dirname, 'node_modules/@supabase/realtime-js'),
  '@supabase/gotrue-js': path.resolve(__dirname, 'node_modules/@supabase/gotrue-js'),
  '@supabase/supabase-js': path.resolve(__dirname, 'node_modules/@supabase/supabase-js'),
};

// Add root node_modules and Supabase packages to watchFolders
config.watchFolders = [
  path.resolve(__dirname, 'node_modules'),
  path.resolve(__dirname, 'node_modules/@supabase/postgrest-js'),
  path.resolve(__dirname, 'node_modules/@supabase/storage-js'),
  path.resolve(__dirname, 'node_modules/@supabase/realtime-js'),
  path.resolve(__dirname, 'node_modules/@supabase/gotrue-js'),
  path.resolve(__dirname, 'node_modules/@supabase/supabase-js'),
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