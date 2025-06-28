import { getApiKey } from './apiKeyManager';

export interface GeminiTranslationRequest {
  text: string;
  fromLanguage: string;
  toLanguage: string;
}

export interface GeminiTranslationResponse {
  translatedText: string;
  confidence?: number;
}

/**
 * Translates text using the Google Gemini API
 */
export async function translateWithGemini(
  request: GeminiTranslationRequest
): Promise<GeminiTranslationResponse> {
  const apiKey = await getApiKey();
  
  if (!apiKey) {
    throw new Error('API key not found. Please configure your Gemini API key in Settings.');
  }

  const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=${apiKey}`;
  
  // Create a specialized prompt for translation
  const prompt = `You are a professional translator specializing in Tamazight (Berber), Arabic, French, and English languages. 

Translate the following text from ${request.fromLanguage} to ${request.toLanguage}:

"${request.text}"

Important guidelines:
- If translating to/from Tamazight, use proper Tifinagh script when appropriate
- Maintain cultural context and nuances
- For emergency or medical terms, prioritize accuracy over literary style
- Return ONLY the translated text, no explanations or additional content
- If the text appears to be a proper noun or untranslatable, keep it as is`;

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{
          parts: [{ text: prompt }]
        }],
        generationConfig: {
          temperature: 0.3, // Lower temperature for more consistent translations
          topK: 40,
          topP: 0.95,
          maxOutputTokens: 1024,
        },
        safetySettings: [
          {
            category: "HARM_CATEGORY_HARASSMENT",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          },
          {
            category: "HARM_CATEGORY_HATE_SPEECH",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          },
          {
            category: "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          },
          {
            category: "HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold: "BLOCK_MEDIUM_AND_ABOVE"
          }
        ]
      })
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => ({}));
      const errorMessage = errorBody.error?.message || `HTTP ${response.status}: ${response.statusText}`;
      
      if (response.status === 400) {
        throw new Error('Invalid API request. Please check your input text.');
      } else if (response.status === 401) {
        throw new Error('Invalid API key. Please check your Gemini API key in Settings.');
      } else if (response.status === 403) {
        throw new Error('API access forbidden. Please verify your API key permissions.');
      } else if (response.status === 429) {
        throw new Error('Rate limit exceeded. Please try again in a moment.');
      } else {
        throw new Error(`API Error: ${errorMessage}`);
      }
    }

    const data = await response.json();
    
    if (!data.candidates || data.candidates.length === 0) {
      throw new Error('No translation generated. Please try again.');
    }

    const translatedText = data.candidates[0].content.parts[0].text.trim();
    
    if (!translatedText) {
      throw new Error('Empty translation received. Please try again.');
    }

    return {
      translatedText,
      confidence: data.candidates[0].finishReason === 'STOP' ? 0.95 : 0.8
    };

  } catch (error: any) {
    console.error('Gemini API translation error:', error);
    
    if (error.message.includes('fetch')) {
      throw new Error('Network error. Please check your internet connection.');
    }
    
    throw error;
  }
}

/**
 * Validates if an API key format looks correct
 */
export function validateApiKeyFormat(apiKey: string): boolean {
  // Gemini API keys typically start with "AIza" and are about 39 characters long
  return /^AIza[A-Za-z0-9_-]{35}$/.test(apiKey.trim());
}

/**
 * Tests the API key by making a simple request
 */
export async function testApiKey(apiKey: string): Promise<boolean> {
  const API_URL = `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=${apiKey}`;
  
  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        contents: [{
          parts: [{ text: 'Hello' }]
        }]
      })
    });

    return response.ok;
  } catch (error) {
    console.error('API key test failed:', error);
    return false;
  }
}