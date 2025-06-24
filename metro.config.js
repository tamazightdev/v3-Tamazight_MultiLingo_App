// Learn more https://docs.expo.io/guides/customizing-metro
const { getDefaultConfig } = require('expo/metro-config');
const path = require('path');

/** @type {import('expo/metro-config').MetroConfig} */
const config = getDefaultConfig(__dirname);

// Add TypeScript extensions to the resolver
config.resolver.sourceExts = ['jsx', 'js', 'ts', 'tsx', 'json'];

// Add explicit module resolution for Supabase dependencies
config.resolver.extraNodeModules = {
  '@supabase/postgrest-js': path.resolve(__dirname, 'node_modules/@supabase/postgrest-js'),
  '@supabase/storage-js': path.resolve(__dirname, 'node_modules/@supabase/storage-js'),
  '@supabase/realtime-js': path.resolve(__dirname, 'node_modules/@supabase/realtime-js'),
  '@supabase/gotrue-js': path.resolve(__dirname, 'node_modules/@supabase/gotrue-js'),
};

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