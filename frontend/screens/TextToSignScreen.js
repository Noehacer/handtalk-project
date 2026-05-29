import React, { useState, useRef } from 'react';
import {
  View, Text, TextInput, TouchableOpacity,
  ScrollView, StyleSheet, ActivityIndicator, Alert,
  Animated, Modal,
} from 'react-native';
import { Image } from 'expo-image';
import * as Clipboard from 'expo-clipboard';
import axios from 'axios';
import { API_URL } from '../config';
import { useTheme } from '../theme';
import { useI18n } from '../i18n';
import { saveToHistory } from '../storage';

const BLURHASH = 'L6PZfSi_.AyE_3t7t7R**0o#DgR4';

export default function TextToSignScreen() {
  const [input, setInput]         = useState('');
  const [signs, setSigns]         = useState([]);
  const [loading, setLoading]     = useState(false);
  const [copied, setCopied]       = useState(false);
  const [selectedSign, setSelectedSign] = useState(null);
  const [foundCount, setFoundCount]     = useState(0);
  const [notFound, setNotFound]         = useState([]);

  const fadeAnim = useRef(new Animated.Value(0)).current;
  const theme = useTheme();
  const t     = useI18n().textToSign;
  const s     = styles(theme);

  const isPhrase = input.trim().includes(' ');
  const wordCount = input.trim().split(' ').filter(Boolean).length;

  const animateIn = () => {
    fadeAnim.setValue(0);
    Animated.timing(fadeAnim, { toValue: 1, duration: 300, useNativeDriver: true }).start();
  };

  const translate = async () => {
    const query = input.trim();
    if (!query) { Alert.alert(t.placeholder); return; }
    setLoading(true);
    setSigns([]);
    setFoundCount(0);
    setNotFound([]);

    try {
      let result = [];
      if (isPhrase) {
        const { data } = await axios.get(
          `${API_URL}/text-to-sign-phrase/${encodeURIComponent(query)}`,
          { timeout: 8000 }
        );
        result = data.signs ?? [];
        setFoundCount(data.found_count || 0);
        setNotFound(data.not_found || []);
      } else {
        const { data } = await axios.get(
          `${API_URL}/text-to-sign/${encodeURIComponent(query)}`,
          { timeout: 8000 }
        );
        result = [data];
        setFoundCount(data.found ? 1 : 0);
        setNotFound(data.found ? [] : [data.word ?? query]);
      }
      setSigns(result);
      animateIn();
      result.forEach((item) => {
        if (item.found) {
          saveToHistory({ type: isPhrase ? 'text_to_sign_phrase' : 'text_to_sign', input: query, output: item.word });
        }
      });
    } catch (err) {
      const msg = !err.response ? t.connectionError : `Error ${err.response.status}`;
      Alert.alert('Error', msg);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = async () => {
    await Clipboard.setStringAsync(input);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const translateLabel = isPhrase
    ? `${t.translatePhrase} (${wordCount} ${t.words})`
    : t.translate;

  return (
    <ScrollView
      style={s.container}
      contentContainerStyle={s.content}
      keyboardShouldPersistTaps="handled"
    >
      <Text style={s.title} accessibilityRole="header">{t.title}</Text>

      <View style={s.inputRow}>
        <TextInput
          style={s.input}
          placeholder={t.placeholder}
          placeholderTextColor={theme.textSecondary}
          value={input}
          onChangeText={setInput}
          onSubmitEditing={translate}
          returnKeyType="search"
          accessibilityLabel={t.placeholder}
          accessibilityHint="Escribe aquí la palabra o frase que quieres traducir a señas"
        />
        {input.length > 0 && (
          <TouchableOpacity
            style={s.iconBtn}
            onPress={handleCopy}
            accessibilityRole="button"
            accessibilityLabel={copied ? 'Copiado' : 'Copiar texto'}
            accessibilityHint="Copia el texto escrito al portapapeles"
          >
            <Text style={{ fontSize: 18 }}>{copied ? '✓' : '📋'}</Text>
          </TouchableOpacity>
        )}
      </View>

      <TouchableOpacity
        style={[s.translateBtn, loading && { opacity: 0.6 }]}
        onPress={translate}
        disabled={loading}
        accessibilityRole="button"
        accessibilityLabel={translateLabel}
        accessibilityHint="Consulta al servidor la imagen de la seña"
      >
        {loading
          ? <ActivityIndicator color="#FFF" accessibilityLabel="Traduciendo..." />
          : <Text style={s.translateBtnText}>{translateLabel}</Text>
        }
      </TouchableOpacity>

      {signs.length > 0 && (
        <Animated.View style={{ opacity: fadeAnim }}>
          {isPhrase ? (
            <View style={s.galleryContainer}>
              <Text style={s.galleryHeader}>
                {foundCount}/{signs.length} señas encontradas
              </Text>
              <ScrollView
                horizontal
                showsHorizontalScrollIndicator={false}
                contentContainerStyle={s.galleryScroll}
              >
                {signs.map((sign, index) => (
                  <TouchableOpacity
                    key={index}
                    style={s.signThumbnail}
                    onPress={() => setSelectedSign(sign)}
                    accessibilityRole="button"
                    accessibilityLabel={`Ver seña para ${sign.word}`}
                  >
                    {sign.found && sign.thumbnail_base64 ? (
                      <Image
                        source={{ uri: sign.thumbnail_base64 }}
                        style={s.thumbnailImg}
                        contentFit="contain"
                      />
                    ) : (
                      <View style={s.thumbnailPlaceholder}>
                        <Text style={s.thumbnailNotFound}>?</Text>
                      </View>
                    )}
                    <Text style={s.thumbnailWord} numberOfLines={1}>
                      {sign.word}
                    </Text>
                    {!sign.found && (
                      <Text style={s.thumbnailUnavailable}>No disponible</Text>
                    )}
                    {sign.found && !sign.has_real_image && (
                      <Text style={s.thumbnailIllustrative}>Ilustrativa</Text>
                    )}
                  </TouchableOpacity>
                ))}
              </ScrollView>
              {notFound.length > 0 && (
                <Text style={s.notFoundText}>
                  Sin seña: {notFound.join(', ')}
                </Text>
              )}
            </View>
          ) : (
            <View style={{ gap: 14 }}>
              {signs.map((item, i) => (
                <SignCard
                  key={i}
                  item={item}
                  theme={theme}
                  styles={s}
                  apiUrl={API_URL}
                  t={t}
                  onSuggestion={(word) => { setInput(word); setSigns([]); }}
                />
              ))}
            </View>
          )}
        </Animated.View>
      )}

      <Modal
        visible={selectedSign !== null}
        transparent
        animationType="fade"
        onRequestClose={() => setSelectedSign(null)}
      >
        <TouchableOpacity
          style={s.modalOverlay}
          activeOpacity={1}
          onPress={() => setSelectedSign(null)}
        >
          {selectedSign?.base64 ? (
            <Image
              source={{ uri: selectedSign.base64 }}
              style={s.modalImage}
              contentFit="contain"
            />
          ) : (
            <View style={s.modalNoImage}>
              <Text style={s.modalNoImageText}>
                Seña no disponible: {selectedSign?.word}
              </Text>
            </View>
          )}
          <Text style={s.modalWord}>{selectedSign?.word}</Text>
          {selectedSign?.category && (
            <Text style={s.modalCategory}>#{selectedSign.category}</Text>
          )}
        </TouchableOpacity>
      </Modal>
    </ScrollView>
  );
}

function SignCard({ item, theme, styles: s, apiUrl, t, onSuggestion }) {
  const isGif = item.media_type === 'image/gif';
  const imgSource = item.base64
    ? { uri: item.base64 }
    : item.path ? { uri: `${apiUrl}${item.path}` } : null;

  return (
    <View
      style={s.signCard}
      accessibilityRole="none"
      accessible
      accessibilityLabel={`Seña para ${item.word?.replace(/_/g, ' ')}${item.category ? `, categoría ${item.category}` : ''}`}
    >
      <Text style={s.wordLabel}>
        {item.word?.replace(/_/g, ' ') ?? '—'}
        {isGif && <Text style={s.gifBadge}> GIF</Text>}
      </Text>

      {item.found && imgSource ? (
        <Image
          source={imgSource}
          style={s.signImage}
          contentFit="contain"
          placeholder={BLURHASH}
          transition={300}
          autoplay={isGif}
          accessibilityLabel={`Imagen de la seña ${item.word?.replace(/_/g, ' ')}`}
          accessibilityRole="image"
        />
      ) : (
        <View style={s.placeholder}>
          <Text style={s.placeholderText}>
            {item.found ? t.imgUnavailable : t.notFound}
          </Text>
          {item.suggestion && (
            <TouchableOpacity
              style={[s.suggestionBtn, { borderColor: theme.primary }]}
              onPress={() => onSuggestion(item.suggestion)}
              accessibilityRole="button"
              accessibilityLabel={`${t.didYouMean} ${item.suggestion.replace(/_/g, ' ')}`}
              accessibilityHint="Toca para buscar esta sugerencia"
            >
              <Text style={[s.suggestionText, { color: theme.primary }]}>
                {t.didYouMean} "{item.suggestion.replace(/_/g, ' ')}"?
              </Text>
            </TouchableOpacity>
          )}
        </View>
      )}

      {item.category && (
        <Text style={s.category} accessibilityLabel={`Categoría: ${item.category}`}>
          {item.category}
        </Text>
      )}
    </View>
  );
}

const styles = (t) => StyleSheet.create({
  container: { flex: 1, backgroundColor: t.background },
  content:   { padding: 20, gap: 14 },
  title:     { fontSize: 28, fontWeight: '800', color: t.text },
  inputRow:  { flexDirection: 'row', alignItems: 'center', gap: 8 },
  input: {
    flex: 1,
    backgroundColor: t.surface,
    borderWidth: 1,
    borderColor: t.border,
    borderRadius: 12,
    padding: 14,
    fontSize: 16,
    color: t.text,
  },
  iconBtn: {
    padding: 12,
    backgroundColor: t.surface,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: t.border,
  },
  translateBtn: { backgroundColor: t.primary, borderRadius: 12, padding: 16, alignItems: 'center' },
  translateBtnText: { color: '#FFF', fontSize: 16, fontWeight: '700' },
  signCard: {
    backgroundColor: t.card,
    borderRadius: 14,
    padding: 16,
    alignItems: 'center',
    gap: 8,
    borderWidth: 1,
    borderColor: t.border,
  },
  wordLabel:    { fontSize: 22, fontWeight: '700', color: t.text, textTransform: 'capitalize' },
  gifBadge:     { fontSize: 12, fontWeight: '600', color: t.primary, textTransform: 'none' },
  signImage:    { width: 240, height: 240 },
  placeholder:  { width: 240, minHeight: 100, backgroundColor: t.surface, borderRadius: 10, justifyContent: 'center', alignItems: 'center', padding: 16, gap: 10 },
  placeholderText: { color: t.textSecondary, fontSize: 14, textAlign: 'center' },
  suggestionBtn:   { borderWidth: 1, borderRadius: 8, paddingVertical: 6, paddingHorizontal: 12 },
  suggestionText:  { fontSize: 14, fontWeight: '600' },
  category:        { fontSize: 12, color: t.textSecondary, textTransform: 'capitalize' },

  // Gallery styles
  galleryContainer:      { marginTop: 16 },
  galleryHeader:         { fontSize: 13, color: '#888', marginBottom: 8, textAlign: 'center' },
  galleryScroll:         { paddingVertical: 4 },
  signThumbnail:         { width: 110, marginRight: 10, alignItems: 'center', backgroundColor: '#f5f5f5', borderRadius: 10, padding: 6 },
  thumbnailImg:          { width: 90, height: 90, borderRadius: 8 },
  thumbnailPlaceholder:  { width: 90, height: 90, borderRadius: 8, backgroundColor: '#ddd', justifyContent: 'center', alignItems: 'center' },
  thumbnailNotFound:     { fontSize: 28, color: '#e74c3c' },
  thumbnailWord:         { fontSize: 12, fontWeight: 'bold', marginTop: 4, textAlign: 'center' },
  thumbnailUnavailable:  { fontSize: 10, color: '#e74c3c', textAlign: 'center' },
  thumbnailIllustrative: { fontSize: 9, color: '#888', textAlign: 'center' },
  notFoundText:          { color: '#e74c3c', fontSize: 12, marginTop: 8, textAlign: 'center' },

  // Modal styles
  modalOverlay:    { flex: 1, backgroundColor: 'rgba(0,0,0,0.85)', justifyContent: 'center', alignItems: 'center' },
  modalImage:      { width: 280, height: 280, borderRadius: 12 },
  modalNoImage:    { width: 280, height: 280, borderRadius: 12, backgroundColor: '#333', justifyContent: 'center', alignItems: 'center', padding: 16 },
  modalNoImageText:{ color: 'white', textAlign: 'center' },
  modalWord:       { color: 'white', fontSize: 22, fontWeight: 'bold', marginTop: 12 },
  modalCategory:   { color: '#aaa', fontSize: 14, marginTop: 4 },
});
