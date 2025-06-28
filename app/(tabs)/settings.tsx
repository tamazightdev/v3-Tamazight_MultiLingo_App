import React, { useState, useEffect } from 'react';
import { View, Text, StyleSheet, SafeAreaView, TouchableOpacity, Animated, TextInput, Alert, ActivityIndicator } from 'react-native';
import { GradientBackground } from '@/components/GradientBackground';
import { GlassCard } from '@/components/GlassCard';
import { Settings, Wifi, WifiOff, Zap, Shield, Globe, KeyRound, Eye, EyeOff, CheckCircle } from 'lucide-react-native';
import { useMode } from '../context/ModeContext';
import { saveApiKey, getApiKey, removeApiKey } from '../services/apiKeyManager';
import { validateApiKeyFormat, testApiKey } from '../services/geminiService';

interface CustomToggleProps {
  value: boolean;
  onValueChange: () => void;
}

const CustomToggle: React.FC<CustomToggleProps> = ({ value, onValueChange }) => {
  const animatedValue = React.useRef(new Animated.Value(value ? 1 : 0)).current;

  const translateX = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: [4, 28],
  });

  const trackAnimatedStyle = {
    backgroundColor: animatedValue.interpolate({
      inputRange: [0, 1],
      outputRange: ['rgba(255, 255, 255, 0.2)', 'rgba(16, 185, 129, 0.8)'],
    }),
  };

  React.useEffect(() => {
    Animated.timing(animatedValue, {
      toValue: value ? 1 : 0,
      duration: 300,
      useNativeDriver: false,
    }).start();
  }, [value, animatedValue]);

  return (
    <TouchableOpacity 
      onPress={onValueChange}
      style={styles.toggleContainer}
      activeOpacity={0.7}
    >
      <Animated.View style={[styles.switchTrack, trackAnimatedStyle]}>
        <Animated.View style={[styles.switchThumb, { transform: [{ translateX }] }]} />
      </Animated.View>
    </TouchableOpacity>
  );
};

export default function SettingsScreen() {
  const { mode, toggleMode } = useMode();
  const isOnline = mode === 'online';

  // API Key management state
  const [apiKeyInput, setApiKeyInput] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [isTestingKey, setIsTestingKey] = useState(false);
  const [keyStatus, setKeyStatus] = useState<'none' | 'valid' | 'invalid'>('none');

  // Load the saved API key when the component mounts
  useEffect(() => {
    const loadKey = async () => {
      const savedKey = await getApiKey();
      if (savedKey) {
        setApiKeyInput(savedKey);
        setKeyStatus('valid');
      }
    };
    loadKey();
  }, []);

  // Handle saving the new key
  const handleSaveApiKey = async () => {
    const trimmedKey = apiKeyInput.trim();
    
    if (!trimmedKey) {
      Alert.alert('Empty Key', 'Please enter a valid Gemini API key.');
      return;
    }

    if (!validateApiKeyFormat(trimmedKey)) {
      Alert.alert(
        'Invalid Format', 
        'The API key format appears incorrect. Gemini API keys should start with "AIza" and be about 39 characters long.'
      );
      return;
    }

    setIsTestingKey(true);
    
    try {
      // Test the API key
      const isValid = await testApiKey(trimmedKey);
      
      if (!isValid) {
        setKeyStatus('invalid');
        Alert.alert(
          'Invalid API Key', 
          'The API key could not be verified. Please check that it\'s correct and has the necessary permissions.'
        );
        setIsTestingKey(false);
        return;
      }

      // Save the key if it's valid
      await saveApiKey(trimmedKey);
      setKeyStatus('valid');
      Alert.alert('Success', 'Your Gemini API key has been saved and verified successfully.');
      
    } catch (error: any) {
      setKeyStatus('invalid');
      Alert.alert('Error', `Failed to verify API key: ${error.message}`);
    } finally {
      setIsTestingKey(false);
    }
  };

  const handleRemoveApiKey = async () => {
    Alert.alert(
      'Remove API Key',
      'Are you sure you want to remove your saved API key? You will need to enter it again to use online mode.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Remove',
          style: 'destructive',
          onPress: async () => {
            try {
              await removeApiKey();
              setApiKeyInput('');
              setKeyStatus('none');
              Alert.alert('Removed', 'Your API key has been removed.');
            } catch (error) {
              Alert.alert('Error', 'Failed to remove API key.');
            }
          }
        }
      ]
    );
  };

  const getKeyStatusColor = () => {
    switch (keyStatus) {
      case 'valid': return '#10B981';
      case 'invalid': return '#EF4444';
      default: return 'rgba(255, 255, 255, 0.6)';
    }
  };

  const getKeyStatusText = () => {
    switch (keyStatus) {
      case 'valid': return 'API key verified ✓';
      case 'invalid': return 'API key invalid ✗';
      default: return 'No API key configured';
    }
  };

  return (
    <View style={styles.container}>
      <GradientBackground>
        <SafeAreaView style={styles.safeArea}>
          <View style={styles.header}>
            <View style={styles.titleRow}>
              <Settings size={32} color="#FFFFFF" strokeWidth={2} />
              <Text style={styles.title}>Settings</Text>
            </View>
            <Text style={styles.subtitle}>Configure your translation preferences</Text>
          </View>

          {/* Translation Mode Card */}
          <GlassCard style={styles.settingsCard}>
            <TouchableOpacity 
              style={styles.settingRow}
              onPress={toggleMode}
              activeOpacity={0.7}
            >
              <View style={styles.settingTextContainer}>
                <View style={styles.labelRow}>
                  <Text style={styles.settingLabel}>Translation Mode</Text>
                  <View style={styles.modeIndicator}>
                    {isOnline ? (
                      <Wifi size={16} color="#10B981" strokeWidth={2} />
                    ) : (
                      <WifiOff size={16} color="rgba(255, 255, 255, 0.6)" strokeWidth={2} />
                    )}
                    <Text style={styles.modeText}>
                      {isOnline ? 'Online' : 'Offline'}
                    </Text>
                  </View>
                </View>
                <Text style={styles.settingDescription}>
                  {isOnline
                    ? 'Online mode uses the Google Gemini API for high-quality translations with internet connection.'
                    : 'Offline mode uses on-device AI and does not require an internet connection for complete privacy.'}
                </Text>
                <TouchableOpacity 
                  style={styles.tapHint}
                  onPress={toggleMode}
                  activeOpacity={0.7}
                >
                  <Text style={styles.tapHintText}>Tap anywhere to toggle</Text>
                </TouchableOpacity>
              </View>
              <CustomToggle value={isOnline} onValueChange={toggleMode} />
            </TouchableOpacity>
          </GlassCard>

          {/* API Key Configuration Card */}
          <GlassCard style={styles.apiKeyCard}>
            <View style={styles.apiKeyHeader}>
              <KeyRound size={20} color="#3B82F6" strokeWidth={2} />
              <Text style={styles.infoTitle}>Google Gemini API Key</Text>
              {keyStatus === 'valid' && (
                <CheckCircle size={16} color="#10B981" strokeWidth={2} />
              )}
            </View>
            
            <Text style={styles.settingDescription}>
              Enter your personal Google AI Gemini API key to use online translation mode. 
              Get your free API key at ai.google.dev.
            </Text>

            <View style={styles.keyStatusContainer}>
              <Text style={[styles.keyStatusText, { color: getKeyStatusColor() }]}>
                {getKeyStatusText()}
              </Text>
            </View>

            <View style={styles.apiKeyInputContainer}>
              <TextInput
                style={styles.apiKeyInput}
                placeholder="Enter your Gemini API key (AIza...)"
                placeholderTextColor="rgba(255, 255, 255, 0.5)"
                value={apiKeyInput}
                onChangeText={setApiKeyInput}
                secureTextEntry={!showApiKey}
                autoCapitalize="none"
                autoCorrect={false}
              />
              <TouchableOpacity 
                style={styles.eyeButton}
                onPress={() => setShowApiKey(!showApiKey)}
              >
                {showApiKey ? (
                  <EyeOff size={20} color="rgba(255, 255, 255, 0.7)" strokeWidth={2} />
                ) : (
                  <Eye size={20} color="rgba(255, 255, 255, 0.7)" strokeWidth={2} />
                )}
              </TouchableOpacity>
            </View>

            <View style={styles.apiKeyActions}>
              <TouchableOpacity 
                style={[styles.saveButton, isTestingKey && styles.saveButtonDisabled]} 
                onPress={handleSaveApiKey}
                disabled={isTestingKey}
              >
                {isTestingKey ? (
                  <ActivityIndicator size="small" color="#FFFFFF" />
                ) : (
                  <Text style={styles.saveButtonText}>
                    {keyStatus === 'valid' ? 'Update Key' : 'Save & Verify'}
                  </Text>
                )}
              </TouchableOpacity>

              {keyStatus === 'valid' && (
                <TouchableOpacity 
                  style={styles.removeButton} 
                  onPress={handleRemoveApiKey}
                >
                  <Text style={styles.removeButtonText}>Remove</Text>
                </TouchableOpacity>
              )}
            </View>

            <Text style={styles.apiKeyNote}>
              💡 Your API key is stored securely on your device and never shared.
            </Text>
          </GlassCard>

          {/* Mode Comparison Card */}
          <GlassCard style={styles.infoCard}>
            <View style={styles.infoHeader}>
              <Zap size={20} color="#F59E0B" strokeWidth={2} />
              <Text style={styles.infoTitle}>Mode Comparison</Text>
            </View>
            
            <View style={styles.comparisonRow}>
              <View style={styles.comparisonColumn}>
                <View style={styles.comparisonHeader}>
                  <Wifi size={16} color="#10B981" strokeWidth={2} />
                  <Text style={styles.comparisonTitle}>Online Mode</Text>
                </View>
                <Text style={styles.comparisonItem}>• Google Gemini AI</Text>
                <Text style={styles.comparisonItem}>• Highest accuracy</Text>
                <Text style={styles.comparisonItem}>• Requires internet</Text>
                <Text style={styles.comparisonItem}>• Requires API key</Text>
              </View>
              
              <View style={styles.comparisonColumn}>
                <View style={styles.comparisonHeader}>
                  <Shield size={16} color="#8B5CF6" strokeWidth={2} />
                  <Text style={styles.comparisonTitle}>Offline Mode</Text>
                </View>
                <Text style={styles.comparisonItem}>• Complete privacy</Text>
                <Text style={styles.comparisonItem}>• No internet needed</Text>
                <Text style={styles.comparisonItem}>• Faster response</Text>
                <Text style={styles.comparisonItem}>• On-device AI</Text>
              </View>
            </View>
          </GlassCard>

          {/* About Card */}
          <GlassCard style={styles.aboutCard}>
            <View style={styles.aboutHeader}>
              <Globe size={20} color="#3B82F6" strokeWidth={2} />
              <Text style={styles.aboutTitle}>About Tamazight Translation</Text>
            </View>
            <Text style={styles.aboutText}>
              This app supports the preservation and promotion of Tamazight (ⵜⴰⵎⴰⵣⵉⵖⵜ), 
              an official language of Morocco. Our AI models are specifically optimized 
              for Moroccan Amazigh variants to ensure cultural and linguistic accuracy.
            </Text>
          </GlassCard>
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
  titleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
    marginBottom: 8,
  },
  title: {
    fontSize: 28,
    fontFamily: 'Inter-Bold',
    color: '#FFFFFF',
  },
  subtitle: {
    fontSize: 16,
    fontFamily: 'Inter-Regular',
    color: 'rgba(255, 255, 255, 0.9)',
    textAlign: 'center',
  },
  settingsCard: {
    marginHorizontal: 20,
    marginBottom: 16,
  },
  settingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  settingTextContainer: {
    flex: 1,
    marginRight: 16,
  },
  labelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  settingLabel: {
    color: '#FFFFFF',
    fontSize: 18,
    fontFamily: 'Inter-SemiBold',
  },
  modeIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  modeText: {
    color: 'rgba(255, 255, 255, 0.9)',
    fontSize: 12,
    fontFamily: 'Inter-Medium',
  },
  settingDescription: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    lineHeight: 20,
    marginBottom: 8,
  },
  tapHint: {
    alignSelf: 'flex-start',
  },
  tapHintText: {
    color: 'rgba(16, 185, 129, 0.8)',
    fontSize: 12,
    fontFamily: 'Inter-Medium',
    fontStyle: 'italic',
  },
  toggleContainer: {
    padding: 8,
  },
  switchTrack: {
    width: 56,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
  },
  switchThumb: {
    width: 24,
    height: 24,
    borderRadius: 12,
    backgroundColor: '#FFFFFF',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 4,
  },
  apiKeyCard: {
    marginHorizontal: 20,
    marginBottom: 16,
  },
  apiKeyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  keyStatusContainer: {
    marginVertical: 8,
  },
  keyStatusText: {
    fontSize: 12,
    fontFamily: 'Inter-Medium',
  },
  apiKeyInputContainer: {
    position: 'relative',
    marginBottom: 16,
  },
  apiKeyInput: {
    backgroundColor: 'rgba(0, 0, 0, 0.2)',
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.3)',
    padding: 12,
    paddingRight: 50,
    color: '#FFFFFF',
    fontFamily: 'Inter-Regular',
    fontSize: 14,
  },
  eyeButton: {
    position: 'absolute',
    right: 12,
    top: 12,
    padding: 4,
  },
  apiKeyActions: {
    flexDirection: 'row',
    gap: 12,
    marginBottom: 12,
  },
  saveButton: {
    backgroundColor: 'rgba(59, 130, 246, 0.8)',
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 20,
    flex: 1,
    alignItems: 'center',
  },
  saveButtonDisabled: {
    backgroundColor: 'rgba(59, 130, 246, 0.4)',
  },
  saveButtonText: {
    color: '#FFFFFF',
    fontFamily: 'Inter-SemiBold',
    fontSize: 14,
  },
  removeButton: {
    backgroundColor: 'rgba(239, 68, 68, 0.8)',
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 20,
    alignItems: 'center',
  },
  removeButtonText: {
    color: '#FFFFFF',
    fontFamily: 'Inter-SemiBold',
    fontSize: 14,
  },
  apiKeyNote: {
    color: 'rgba(255, 255, 255, 0.7)',
    fontSize: 12,
    fontFamily: 'Inter-Regular',
    fontStyle: 'italic',
  },
  infoCard: {
    marginHorizontal: 20,
    marginBottom: 16,
  },
  infoHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 16,
  },
  infoTitle: {
    color: '#FFFFFF',
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
  },
  comparisonRow: {
    flexDirection: 'row',
    gap: 16,
  },
  comparisonColumn: {
    flex: 1,
  },
  comparisonHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 6,
    marginBottom: 8,
  },
  comparisonTitle: {
    color: '#FFFFFF',
    fontSize: 14,
    fontFamily: 'Inter-SemiBold',
  },
  comparisonItem: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 12,
    fontFamily: 'Inter-Regular',
    marginBottom: 4,
  },
  aboutCard: {
    marginHorizontal: 20,
  },
  aboutHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 12,
  },
  aboutTitle: {
    color: '#FFFFFF',
    fontSize: 16,
    fontFamily: 'Inter-SemiBold',
  },
  aboutText: {
    color: 'rgba(255, 255, 255, 0.8)',
    fontSize: 14,
    fontFamily: 'Inter-Regular',
    lineHeight: 20,
  },
});