import { Platform } from 'react-native';
import { supabase } from './supabaseClient';

// Define the structure of a translation item
export interface TranslationItem {
  id: string;
  sourceText: string;
  translatedText: string;
  fromLang: string;
  toLang: string;
  timestamp: string; // Use ISO string for consistency
  isFavorite: boolean;
}

// --- Web Storage (IndexedDB simulation) ---
const webStorage = {
  getHistory: (): Promise<TranslationItem[]> => {
    return new Promise((resolve) => {
      const stored = localStorage.getItem('tamazight_translations');
      const data = stored ? JSON.parse(stored) : [];
      resolve(data.sort((a: TranslationItem, b: TranslationItem) => 
        new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
      ));
    });
  },
  
  addTranslation: (item: Omit<TranslationItem, 'id' | 'timestamp'>): Promise<void> => {
    return new Promise((resolve) => {
      const stored = localStorage.getItem('tamazight_translations');
      const data = stored ? JSON.parse(stored) : [];
      
      const newItem: TranslationItem = {
        ...item,
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
      };
      
      data.push(newItem);
      localStorage.setItem('tamazight_translations', JSON.stringify(data));
      resolve();
    });
  },
  
  toggleFavorite: (id: string): Promise<void> => {
    return new Promise((resolve) => {
      const stored = localStorage.getItem('tamazight_translations');
      const data = stored ? JSON.parse(stored) : [];
      
      const item = data.find((t: TranslationItem) => t.id === id);
      if (item) {
        item.isFavorite = !item.isFavorite;
        localStorage.setItem('tamazight_translations', JSON.stringify(data));
      }
      resolve();
    });
  },
  
  deleteTranslation: (id: string): Promise<void> => {
    return new Promise((resolve) => {
      const stored = localStorage.getItem('tamazight_translations');
      const data = stored ? JSON.parse(stored) : [];
      
      const filtered = data.filter((t: TranslationItem) => t.id !== id);
      localStorage.setItem('tamazight_translations', JSON.stringify(filtered));
      resolve();
    });
  },
};

// --- Better-SQLite3 Database (Mobile) ---
let db: any = null;

const initMobileDB = async () => {
  if (Platform.OS === 'web') return null;
  
  try {
    // Dynamic import for mobile platforms only
    const Database = require('better-sqlite3');
    db = new Database('tamazight_translations.db');
    
    // Initialize the database table
    db.exec(`
      CREATE TABLE IF NOT EXISTS translations (
        id TEXT PRIMARY KEY,
        sourceText TEXT NOT NULL,
        translatedText TEXT NOT NULL,
        fromLang TEXT NOT NULL,
        toLang TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        isFavorite INTEGER NOT NULL
      );
    `);
    
    return db;
  } catch (error) {
    console.warn('Better-SQLite3 not available, falling back to web storage');
    return null;
  }
};

const mobileStorage = {
  getHistory: async (): Promise<TranslationItem[]> => {
    if (!db) await initMobileDB();
    if (!db) return webStorage.getHistory();
    
    const stmt = db.prepare('SELECT * FROM translations ORDER BY timestamp DESC');
    return stmt.all().map((item: any) => ({
      ...item,
      isFavorite: Boolean(item.isFavorite),
    })) as TranslationItem[];
  },
  
  addTranslation: async (item: Omit<TranslationItem, 'id' | 'timestamp'>): Promise<void> => {
    if (!db) await initMobileDB();
    if (!db) return webStorage.addTranslation(item);
    
    const id = crypto.randomUUID();
    const timestamp = new Date().toISOString();
    const stmt = db.prepare(`
      INSERT INTO translations (id, sourceText, translatedText, fromLang, toLang, timestamp, isFavorite)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    stmt.run(id, item.sourceText, item.translatedText, item.fromLang, item.toLang, timestamp, Number(item.isFavorite));
  },
  
  toggleFavorite: async (id: string): Promise<void> => {
    if (!db) await initMobileDB();
    if (!db) return webStorage.toggleFavorite(id);
    
    const stmt = db.prepare('UPDATE translations SET isFavorite = NOT isFavorite WHERE id = ?');
    stmt.run(id);
  },
  
  deleteTranslation: async (id: string): Promise<void> => {
    if (!db) await initMobileDB();
    if (!db) return webStorage.deleteTranslation(id);
    
    const stmt = db.prepare('DELETE FROM translations WHERE id = ?');
    stmt.run(id);
  },
};

// --- Online Database (Supabase Edge Functions) ---
const onlineDB = {
  getHistory: async (): Promise<TranslationItem[]> => {
    if (!supabase) {
      throw new Error('Supabase not configured. Please add EXPO_PUBLIC_SUPABASE_URL and EXPO_PUBLIC_SUPABASE_ANON_KEY to your environment variables.');
    }
    
    const { data, error } = await supabase.functions.invoke('translation-history', {
      method: 'GET',
    });
    if (error) throw error;
    return data.history || [];
  },
  
  addTranslation: async (item: Omit<TranslationItem, 'id' | 'timestamp'>): Promise<void> => {
    if (!supabase) {
      throw new Error('Supabase not configured. Please add EXPO_PUBLIC_SUPABASE_URL and EXPO_PUBLIC_SUPABASE_ANON_KEY to your environment variables.');
    }
    
    const { error } = await supabase.functions.invoke('translation-history', {
      method: 'POST',
      body: { item },
    });
    if (error) throw error;
  },
  
  toggleFavorite: async (id: string): Promise<void> => {
    if (!supabase) {
      throw new Error('Supabase not configured. Please add EXPO_PUBLIC_SUPABASE_URL and EXPO_PUBLIC_SUPABASE_ANON_KEY to your environment variables.');
    }
    
    const { error } = await supabase.functions.invoke('translation-history', {
      method: 'PUT',
      body: { id },
    });
    if (error) throw error;
  },
  
  deleteTranslation: async (id: string): Promise<void> => {
    if (!supabase) {
      throw new Error('Supabase not configured. Please add EXPO_PUBLIC_SUPABASE_URL and EXPO_PUBLIC_SUPABASE_ANON_KEY to your environment variables.');
    }
    
    const { error } = await supabase.functions.invoke('translation-history', {
      method: 'DELETE',
      body: { id },
    });
    if (error) throw error;
  },
};

// --- Unified Database Service ---
export const databaseService = {
  getHistory: (mode: 'online' | 'offline'): Promise<TranslationItem[]> => {
    if (mode === 'online') {
      return onlineDB.getHistory();
    } else {
      return Platform.OS === 'web' ? webStorage.getHistory() : mobileStorage.getHistory();
    }
  },
  
  addTranslation: (mode: 'online' | 'offline', item: Omit<TranslationItem, 'id' | 'timestamp'>): Promise<void> => {
    if (mode === 'online') {
      return onlineDB.addTranslation(item);
    } else {
      return Platform.OS === 'web' ? webStorage.addTranslation(item) : mobileStorage.addTranslation(item);
    }
  },
  
  toggleFavorite: (mode: 'online' | 'offline', id: string): Promise<void> => {
    if (mode === 'online') {
      return onlineDB.toggleFavorite(id);
    } else {
      return Platform.OS === 'web' ? webStorage.toggleFavorite(id) : mobileStorage.toggleFavorite(id);
    }
  },
  
  deleteTranslation: (mode: 'online' | 'offline', id: string): Promise<void> => {
    if (mode === 'online') {
      return onlineDB.deleteTranslation(id);
    } else {
      return Platform.OS === 'web' ? webStorage.deleteTranslation(id) : mobileStorage.deleteTranslation(id);
    }
  },
};