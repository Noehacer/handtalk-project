import React, { useState, useCallback } from 'react';
import {
  View, Text, FlatList, TouchableOpacity,
  StyleSheet, Alert,
} from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { loadHistory, clearHistory } from '../storage';
import { useTheme } from '../theme';
import { useI18n } from '../i18n';

const TYPE_LABEL = {
  sign_to_text:         '🤟 → Texto',
  sign_to_text_dynamic: '🤟 → Texto (dinámica)',
  text_to_sign:         'Texto → 🤟',
  text_to_sign_phrase:  'Frase → 🤟',
};

function formatDate(iso) {
  return new Date(iso).toLocaleString('es-MX', { dateStyle: 'short', timeStyle: 'short' });
}

export default function HistoryScreen() {
  const [history, setHistory] = useState([]);
  const theme = useTheme();
  const t     = useI18n().history;
  const s     = styles(theme);

  useFocusEffect(
    useCallback(() => { loadHistory().then(setHistory); }, [])
  );

  const handleClear = () => {
    Alert.alert(t.clearHistory, t.clearConfirm, [
      { text: t.cancel, style: 'cancel' },
      {
        text: t.delete,
        style: 'destructive',
        onPress: async () => { await clearHistory(); setHistory([]); },
      },
    ]);
  };

  if (history.length === 0) {
    return (
      <View
        style={s.empty}
        accessible
        accessibilityLabel={`${t.empty}. ${t.emptySub}`}
      >
        <Text style={s.emptyText}>{t.empty}</Text>
        <Text style={s.emptySubtext}>{t.emptySub}</Text>
      </View>
    );
  }

  return (
    <View style={s.container}>
      <FlatList
        data={history}
        keyExtractor={(_, i) => String(i)}
        contentContainerStyle={{ padding: 12 }}
        ListHeaderComponent={
          <TouchableOpacity
            style={s.clearBtn}
            onPress={handleClear}
            accessibilityRole="button"
            accessibilityLabel={t.clearHistory}
            accessibilityHint="Elimina todas las traducciones del historial"
          >
            <Text style={s.clearBtnText}>{t.clearHistory}</Text>
          </TouchableOpacity>
        }
        renderItem={({ item }) => (
          <HistoryCard item={item} theme={theme} styles={s} t={t} />
        )}
      />
    </View>
  );
}

function HistoryCard({ item, styles: s, t }) {
  const confPct = item.confidence > 0
    ? `${t.confidence}: ${(item.confidence * 100).toFixed(0)}%`
    : null;

  const a11yLabel = [
    TYPE_LABEL[item.type] ?? item.type,
    item.output,
    item.input && !item.input.startsWith('[') ? `"${item.input}"` : null,
    confPct,
    formatDate(item.date),
  ].filter(Boolean).join('. ');

  return (
    <View
      style={s.card}
      accessible
      accessibilityRole="text"
      accessibilityLabel={a11yLabel}
    >
      <View style={s.cardHeader}>
        <Text style={s.typeTag}>{TYPE_LABEL[item.type] ?? item.type}</Text>
        <Text style={s.date}>{formatDate(item.date)}</Text>
      </View>
      <Text style={s.output}>{item.output}</Text>
      {item.input && !item.input.startsWith('[') && (
        <Text style={s.input}>"{item.input}"</Text>
      )}
      {confPct && <Text style={s.confidence}>{confPct}</Text>}
    </View>
  );
}

const styles = (t) => StyleSheet.create({
  container: { flex: 1, backgroundColor: t.background },
  empty: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: t.background, gap: 8 },
  emptyText:    { fontSize: 18, fontWeight: '600', color: t.text },
  emptySubtext: { fontSize: 14, color: t.textSecondary },
  clearBtn: { backgroundColor: t.error, padding: 10, borderRadius: 8, alignItems: 'center', marginBottom: 12 },
  clearBtnText: { color: '#FFF', fontWeight: 'bold', fontSize: 14 },
  card: {
    backgroundColor: t.card,
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderLeftWidth: 4,
    borderLeftColor: t.primary,
  },
  cardHeader:  { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 6 },
  typeTag:     { fontSize: 12, fontWeight: '700', color: t.primary },
  date:        { fontSize: 11, color: t.textSecondary },
  output:      { fontSize: 20, fontWeight: 'bold', color: t.text },
  input:       { fontSize: 13, color: t.textSecondary, marginTop: 3 },
  confidence:  { fontSize: 12, color: t.textSecondary, marginTop: 3 },
});
