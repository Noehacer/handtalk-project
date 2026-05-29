import React, { useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Animated } from 'react-native';

export default function Logo({ size = 80, color = '#007AFF', animated = false }) {
  const scaleAnim = useRef(new Animated.Value(animated ? 0 : 1)).current;
  const opacityAnim = useRef(new Animated.Value(animated ? 0 : 1)).current;

  useEffect(() => {
    if (!animated) return;
    Animated.parallel([
      Animated.spring(scaleAnim,   { toValue: 1, friction: 5, tension: 80, useNativeDriver: true }),
      Animated.timing(opacityAnim, { toValue: 1, duration: 400, useNativeDriver: true }),
    ]).start();
  }, [animated]);

  const radius    = size / 4;
  const fontSize  = size * 0.34;
  const handSize  = size * 0.28;

  return (
    <Animated.View style={{ transform: [{ scale: scaleAnim }], opacity: opacityAnim, alignItems: 'center' }}>
      <View style={[styles.circle, { width: size, height: size, borderRadius: radius, backgroundColor: color }]}>
        {/* Mano estilizada con texto */}
        <Text style={[styles.hand, { fontSize: handSize }]}>🤟</Text>
      </View>
      <View style={[styles.badge, { backgroundColor: color }]}>
        <Text style={[styles.badgeText, { fontSize: fontSize * 0.45 }]}>HandTalk</Text>
      </View>
    </Animated.View>
  );
}

const styles = StyleSheet.create({
  circle: {
    justifyContent: 'center',
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 8,
  },
  hand: {
    lineHeight: undefined,
  },
  badge: {
    marginTop: 6,
    paddingHorizontal: 12,
    paddingVertical: 3,
    borderRadius: 20,
  },
  badgeText: {
    color: '#FFF',
    fontWeight: '800',
    letterSpacing: 0.5,
  },
});
