import React, { useState, useEffect, useCallback } from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, SafeAreaView, ActivityIndicator } from 'react-native';
import { useFocusEffect } from 'expo-router';
import { GradientBackground } from '@/components/GradientBackground';
import { GlassCard } from '@/components/GlassCard';
import { Search, Star, Trash2, ArrowUpDown, Database, Cloud } from 'lucide-react-native';
import { useMode } from '../context/ModeContext';
import { databaseService, TranslationItem } from '../services/database';

export default function HistoryScreen() {
  const [searchText, setSearchText] = useState('');
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
  const [history, setHistory] = useState<TranslationItem[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const { mode } = useMode();

  const fetchHistory = useCallback(async () => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await databaseService.getHistory(mode);
      setHistory(data);
    } catch (error) {
      console.error('Failed to fetch history:', error);
      setError(mode === 'online' 
        ? 'Failed to fetch online history. Please check your connection.' 
        : 'Failed to fetch local history.');
      setHistory([]);
    } finally {
      setIsLoading(false);
    }
  }, [mode]);

  useFocusEffect(
    useCallback(() => {
      fetchHistory();
    }, [fetchHistory])
  );

  const filteredHistory = history.filter(item => {
    const matchesSearch = searchText === '' || 
      item.sourceText.toLowerCase().includes(searchText.toLowerCase()) ||
      item.translatedText.toLowerCase().includes(searchText.toLowerCase());
    
    const matchesFavorites = !showFavoritesOnly || item.isFavorite;
    
    return matchesSearch && matchesFavorites;
  });

  const toggleFavorite = async (id: string) => {
    try {
      await databaseService.toggleFavorite(mode, id);
      fetchHistory(); // Re-fetch to update the UI
    } catch (error) {
      console.error('Failed to toggle favorite:', error);
    }
  };

  const deleteItem = async (id: string) => {
    try {
      await databaseService.deleteTranslation(mode, id);
      fetchHistory(); // Re-fetch to update the UI
    } catch (error) {
      console.error('Failed to delete translation:', error);
    }
  };

  const formatTimestamp = (timestamp: string) => {
    const date = new Date(timestamp);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <View style={styles.container}>
      <GradientBackground>
        <SafeAreaView style={styles.safeArea}>
          <View style={styles.header}>
            <Text style={styles.title}>Translation History</Text>
            <View style={styles.modeIndicator}>
              {mode === 'online' ? (
                <Cloud size={16} color="#10B981" strokeWidth={2} />
              ) : (
                <Database size={16} color="#8B5CF6" strokeWidth={2} />
              )}
              <Text style={styles.subtitle}>
                {mode === 'online' ? 'Cloud Storage' : 'Local Storage'} • {history.length} translations
              </Text>
            </View>
          </View>

          <GlassCard style={styles.searchCard}>
            <View style={styles.searchContainer}>
              <Search size={20} color="rgba(255, 255, 255, 0.7)" strokeWidth={2} />
              <Text style={styles.searchPlaceholder}>Search translations...</Text>
            </View>
            <TouchableOpacity 
              style={[styles.filterButton, showFavoritesOnly && styles.filterActive]}
              onPress={() => setShowFavoritesOnly(!showFavoritesOnly)}
            >
              <Star size={20} color="#FFFFFF" strokeWidth={2} />
            </TouchableOpacity>
          </GlassCard>

          <ScrollView style={styles.scrollView} showsVerticalScrollIndicator={false}>
            {isLoading ? (
              <View style={styles.loadingContainer}>
                <ActivityIndicator size="large" color="#FFFFFF" />
                <Text style={styles.loadingText}>
                  Loading {mode === 'online' ? 'cloud' : 'local'} history...
                </Text>
              </View>
            ) : error ? (
              <GlassCard style={styles.errorCard}>
                <Text style={styles.errorText}>⚠️ {error}</Text>
                <TouchableOpacity style={styles.retryButton} onPress={fetchHistory}>
                  <Text style={styles.retryText}>Retry</Text>
                </TouchableOpacity>
              </GlassCard>
            ) : filteredHistory.length === 0 ? (
              <GlassCard style={styles.emptyCard}>
                <Text style={styles.emptyText}>No translations found</Text>
                <Text style={styles.emptySubtext}>
                  {showFavoritesOnly 
                    ? 'No favorite translations yet' 
                    : history.length === 0
                      ? 'Start translating to build your history'
                      : 'No translations match your search'}
                </Text>
              </GlassCard>
            ) : (
              filteredHistory.map((item) => (
                <GlassCard key={item.id} style={styles.historyItem}>
                  <View style={styles.itemHeader}>
                    <View style={styles.languageInfo}>
                      <Text style={styles.languageText}>{item.fromLang}</Text>
                      <ArrowUpDown size={16} color="rgba(255, 255, 255, 0.6)" strokeWidth={2} />
                      <Text style={styles.languageText}>{item.toLang}</Text>
                    </View>
                    <View style={styles.itemActions}>
                      <TouchableOpacity 
                        style={styles.actionButton}
                        onPress={() => toggleFavorite(item.id)}
                      >
                        <Star 
                          size={20} 
                          color={item.isFavorite ? '#F59E0B' : 'rgba(255, 255, 255, 0.6)'} 
                          fill={item.isFavorite ? '#F59E0B' : 'none'}
                          strokeWidth={2} 
                        />
                      </TouchableOpacity>
                      <TouchableOpacity 
                        style={styles.actionButton}
                        onPress={() => deleteItem(item.id)}
                      >
                        <Trash2 size={20} color="rgba(239, 68, 68, 0.8)" strokeWidth={2} />
                      </TouchableOpacity>
                    </View>
                  </View>
                  
                  <View style={styles.textContainer}>
                    <Text style={styles.sourceText}>{item.sourceText}</Text>
                    <View style={styles.arrow}>
                      <Text style={styles.arrowText}>↓</Text>
                    </View>
                    <Text style={styles.translatedText}>{item.translatedText}</Text>
                  </View>
                  
                  <Text style={styles.timestamp}>{formatTimestamp(item.timestamp)}</Text>
                </GlassCard>
              ))
            )}
          </ScrollView>
        </SafeAreaView>
      </GradientBackground>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  safeArea: {
    flex: 1,
  },
  header: {
    alignItems: 'center',
    marginBottom: 24,
    marginTop: 20,
    paddingHorizontal: 20,
  },
  title: {
    fontSize: 28,
    fontFamily: 'Inter-Bold',
    color: '#FFFFFF',
    textAlign: 'center',
  },
  modeIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 8,
  },
  subtitle: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: 'rgba(255, 255, 255, 0.8)',
  },
  searchCard: {
    marginHorizontal: 20,
    marginBottom: 16,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    flex: 1,
  },
  searchPlaceholder: {
    color: 'rgba(255, 255, 255, 0.7)',
    fontSize: 16,
    fontFamily: 'Inter-Regular',
  },
  filterButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: 'rgba(255, 255, 255, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    position: 'absolute',
    right: 20,
    top: 20,
  },
  filterActive: {
    backgroundColor: 'rgba(245, 158, 11, 0.8)',
  },
  scrollView: {
    flex: 1,
    paddingHorizontal: 20,
  },
  loadingContainer: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  loadingText: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    marginTop: 16,
  },
  errorCard: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  errorText: {
    color: '#EF4444',
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
    marginBottom: 16,
    textAlign: 'center',
  },
  retryButton: {
    backgroundColor: 'rgba(59, 130, 246, 0.8)',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 12,
  },
  retryText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontFamily: 'Inter-SemiBold',
  },
  historyItem: {
    marginBottom: 16,
  },
  itemHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  languageInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  languageText: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
    fontFamily: 'Inter-Medium',
  },
  itemActions: {
    flexDirection: 'row',
    gap: 8,
  },
  actionButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(255, 255, 255, 0.1)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  textContainer: {
    marginBottom: 12,
  },
  sourceText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    lineHeight: 24,
  },
  arrow: {
    alignItems: 'center',
    marginVertical: 8,
  },
  arrowText: {
    color: 'rgba(255, 255, 255, 0.6)',
    fontSize: 16,
  },
  translatedText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontFamily: 'Inter-Medium',
    lineHeight: 24,
  },
  timestamp: {
    color: 'rgba(255, 255, 255, 0.5)',
    fontSize: 12,
    fontFamily: 'Inter-Regular',
  },
  emptyCard: {
    alignItems: 'center',
    paddingVertical: 40,
  },
  emptyText: {
    color: '#FFFFFF',
    fontSize: 18,
    fontFamily: 'Inter-SemiBold',
    marginBottom: 8,
  },
  emptySubtext: {
    color: 'rgba(255, 255, 255, 0.7)',
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    textAlign: 'center',
  },
});