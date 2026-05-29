import React, { useRef, useState } from 'react';
import {
  View, Text, FlatList, TouchableOpacity,
  StyleSheet, Dimensions, Animated,
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Logo from '../components/Logo';
import { useTheme } from '../theme';
import { useI18n } from '../i18n';

const { width: SCREEN_W } = Dimensions.get('window');
const ONBOARDING_KEY = 'handtalk_onboarding';

export async function markOnboardingSeen() {
  await AsyncStorage.setItem(ONBOARDING_KEY, 'true');
}

export async function isOnboardingSeen() {
  const val = await AsyncStorage.getItem(ONBOARDING_KEY);
  return val === 'true';
}

export default function OnboardingScreen({ onDone }) {
  const [activeIndex, setActiveIndex] = useState(0);
  const flatListRef = useRef(null);
  const theme = useTheme();
  const t = useI18n().onboarding;
  const s = styles(theme);

  const slides = t.slides;
  const isLast = activeIndex === slides.length - 1;

  const handleNext = () => {
    if (isLast) {
      handleDone();
    } else {
      flatListRef.current?.scrollToIndex({ index: activeIndex + 1, animated: true });
    }
  };

  const handleDone = async () => {
    await markOnboardingSeen();
    onDone();
  };

  const onViewableItemsChanged = useRef(({ viewableItems }) => {
    if (viewableItems[0]) setActiveIndex(viewableItems[0].index);
  }).current;

  return (
    <View style={s.container}>
      {/* Skip button */}
      {!isLast && (
        <TouchableOpacity
          style={s.skipBtn}
          onPress={handleDone}
          accessibilityRole="button"
          accessibilityLabel={t.skip}
        >
          <Text style={s.skipText}>{t.skip}</Text>
        </TouchableOpacity>
      )}

      <FlatList
        ref={flatListRef}
        data={slides}
        horizontal
        pagingEnabled
        showsHorizontalScrollIndicator={false}
        keyExtractor={(_, i) => String(i)}
        onViewableItemsChanged={onViewableItemsChanged}
        viewabilityConfig={{ viewAreaCoveragePercentThreshold: 50 }}
        renderItem={({ item, index }) => (
          <Slide
            item={item}
            index={index}
            theme={theme}
            styles={s}
            isFirst={index === 0}
          />
        )}
      />

      {/* Dots */}
      <View style={s.dotsRow} accessibilityRole="progressbar">
        {slides.map((_, i) => (
          <View
            key={i}
            style={[s.dot, i === activeIndex && s.dotActive]}
          />
        ))}
      </View>

      {/* Next / Start */}
      <TouchableOpacity
        style={[s.nextBtn, { backgroundColor: theme.primary }]}
        onPress={handleNext}
        accessibilityRole="button"
        accessibilityLabel={isLast ? t.start : t.next}
        accessibilityHint={isLast ? 'Abre la aplicación' : 'Avanza al siguiente paso'}
      >
        <Text style={s.nextBtnText}>
          {isLast ? t.start : t.next}
        </Text>
      </TouchableOpacity>

      <View style={{ height: 32 }} />
    </View>
  );
}

function Slide({ item, index, theme, styles: s, isFirst }) {
  const fadeAnim = useRef(new Animated.Value(0)).current;

  React.useEffect(() => {
    Animated.timing(fadeAnim, { toValue: 1, duration: 500, delay: 100, useNativeDriver: true }).start();
  }, []);

  return (
    <Animated.View style={[s.slide, { opacity: fadeAnim }]}>
      {index === 0 ? (
        <Logo size={100} color={theme.primary} animated />
      ) : (
        <Text style={s.emoji} accessibilityLabel="">{item.emoji}</Text>
      )}
      <Text style={s.slideTitle} accessibilityRole="header">{item.title}</Text>
      <Text style={s.slideDesc}>{item.description}</Text>
    </Animated.View>
  );
}

const styles = (t) => StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: t.background,
    alignItems: 'center',
  },
  skipBtn: {
    alignSelf: 'flex-end',
    padding: 16,
    marginTop: 8,
  },
  skipText: {
    color: t.textSecondary,
    fontSize: 15,
  },
  slide: {
    width: SCREEN_W,
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 40,
    gap: 24,
  },
  emoji: {
    fontSize: 90,
    textAlign: 'center',
  },
  slideTitle: {
    fontSize: 28,
    fontWeight: '800',
    color: t.text,
    textAlign: 'center',
  },
  slideDesc: {
    fontSize: 16,
    color: t.textSecondary,
    textAlign: 'center',
    lineHeight: 24,
  },
  dotsRow: {
    flexDirection: 'row',
    gap: 8,
    marginVertical: 20,
  },
  dot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: '#CCC',
  },
  dotActive: {
    width: 24,
    backgroundColor: '#007AFF',
  },
  nextBtn: {
    width: SCREEN_W - 48,
    borderRadius: 16,
    padding: 18,
    alignItems: 'center',
  },
  nextBtnText: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: '700',
  },
});
