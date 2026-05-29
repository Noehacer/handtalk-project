import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, StatusBar, SafeAreaView } from 'react-native';
import { useTheme } from '../theme';
import { useI18n } from '../i18n';
import Logo from '../components/Logo';

export default function HomeScreen({ navigation }) {
  const theme = useTheme();
  const t = useI18n().home;
  const s = styles(theme);

  const buttons = [
    {
      key: 'SignToText',
      label: t.signsToText,
      sub: t.signsToTextSub,
      emoji: '🤟',
      bg: theme.primary,
      a11yHint: 'Abre la cámara para detectar señas',
    },
    {
      key: 'TextToSign',
      label: t.textToSigns,
      sub: t.textToSignsSub,
      emoji: '⌨️',
      bg: theme.success,
      a11yHint: 'Abre la pantalla para escribir y traducir palabras a señas',
    },
    {
      key: 'History',
      label: t.history,
      sub: t.historySub,
      emoji: '🕐',
      bg: theme.surface,
      textColor: theme.text,
      border: true,
      a11yHint: 'Ver el historial de traducciones realizadas',
    },
  ];

  return (
    <SafeAreaView style={s.safe}>
      <StatusBar barStyle={theme.text === '#1A1A1A' ? 'dark-content' : 'light-content'} />
      <View style={s.container}>
        <View style={s.header}>
          <Logo size={72} color={theme.primary} animated />
          <Text style={s.subtitle} accessibilityRole="text">{t.subtitle}</Text>
        </View>

        <View style={s.buttons}>
          {buttons.map(({ key, label, sub, emoji, bg, textColor, border, a11yHint }) => (
            <TouchableOpacity
              key={key}
              style={[
                s.btn,
                { backgroundColor: bg },
                border && { borderWidth: 1, borderColor: theme.border },
              ]}
              onPress={() => navigation.navigate(key)}
              accessibilityRole="button"
              accessibilityLabel={label}
              accessibilityHint={a11yHint}
            >
              <Text style={s.btnEmoji} accessibilityElementsHidden>{emoji}</Text>
              <Text style={[s.btnLabel, textColor && { color: textColor }]}>{label}</Text>
              <Text style={[s.btnSub, textColor && { color: theme.textSecondary }]}>{sub}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>
    </SafeAreaView>
  );
}

const styles = (t) => StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: t.background,
  },
  container: {
    flex: 1,
    paddingHorizontal: 24,
    justifyContent: 'center',
    gap: 32,
  },
  header: {
    alignItems: 'center',
    gap: 10,
  },
  subtitle: {
    fontSize: 15,
    color: t.textSecondary,
    textAlign: 'center',
  },
  buttons: {
    gap: 14,
  },
  btn: {
    borderRadius: 18,
    padding: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    elevation: 3,
  },
  btnEmoji: {
    fontSize: 28,
    marginBottom: 4,
  },
  btnLabel: {
    fontSize: 18,
    fontWeight: '700',
    color: '#FFF',
  },
  btnSub: {
    fontSize: 13,
    color: 'rgba(255,255,255,0.8)',
    marginTop: 2,
  },
});
