import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';

const API_KEY_STORAGE_KEY = 'user_gemini_api_key';

/**
 * Saves the user's Gemini API key securely.
 * @param apiKey The API key to save.
 */
export async function saveApiKey(apiKey: string): Promise<void> {
  try {
    if (Platform.OS === 'web') {
      // For web, use localStorage as SecureStore is not available
      localStorage.setItem(API_KEY_STORAGE_KEY, apiKey);
    } else {
      // For mobile, use SecureStore
      await SecureStore.setItemAsync(API_KEY_STORAGE_KEY, apiKey);
    }
    console.log('API Key saved successfully.');
  } catch (error) {
    console.error('Failed to save API key:', error);
    throw new Error('Could not save the API key.');
  }
}

/**
 * Retrieves the user's saved Gemini API key.
 * @returns The saved API key, or null if it doesn't exist.
 */
export async function getApiKey(): Promise<string | null> {
  try {
    if (Platform.OS === 'web') {
      // For web, use localStorage
      return localStorage.getItem(API_KEY_STORAGE_KEY);
    } else {
      // For mobile, use SecureStore
      const apiKey = await SecureStore.getItemAsync(API_KEY_STORAGE_KEY);
      return apiKey;
    }
  } catch (error) {
    console.error('Failed to retrieve API key:', error);
    return null;
  }
}

/**
 * Removes the saved API key.
 */
export async function removeApiKey(): Promise<void> {
  try {
    if (Platform.OS === 'web') {
      localStorage.removeItem(API_KEY_STORAGE_KEY);
    } else {
      await SecureStore.deleteItemAsync(API_KEY_STORAGE_KEY);
    }
    console.log('API Key removed successfully.');
  } catch (error) {
    console.error('Failed to remove API key:', error);
    throw new Error('Could not remove the API key.');
  }
}