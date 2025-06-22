import React from 'react';
import { View, Text, StyleSheet, SafeAreaView, TouchableOpacity, Animated } from 'react-native';
import { GradientBackground } from '@/components/GradientBackground';
import { GlassCard } from '@/components/GlassCard';
import { Settings, Wifi, WifiOff, Zap, Shield, Globe } from 'lucide-react-native';
import { useMode } from '../context/ModeContext';

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
                    ? 'Online mode uses the Gemma-3 API for the highest quality translations with access to the latest language models.'
                    : 'Offline mode uses the on-device AI model and does not require an internet connection for complete privacy.'}
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
                <Text style={styles.comparisonItem}>• Latest AI models</Text>
                <Text style={styles.comparisonItem}>• Highest accuracy</Text>
                <Text style={styles.comparisonItem}>• Requires internet</Text>
                <Text style={styles.comparisonItem}>• Cloud processing</Text>
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

          <GlassCard style={styles.aboutCard}>
            <View style={styles.aboutHeader}>
              <Globe size={20} color="#3B82F6" strokeWidth={2} />
              <Text style={styles.aboutTitle}>About Tamazight Translation</Text>
            </View>
            <Text style={styles.aboutText}>
              This app supports the preservation and promotion of Tamazight (ⵜⴰⵎⴰⵣⵉⵖⵜ), 
              an official language of Morocco. Our AI models are specifically fine-tuned 
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