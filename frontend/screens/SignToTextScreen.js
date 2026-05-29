import React, { useRef, useState, useEffect, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, Switch, StyleSheet,
  ActivityIndicator, Animated, Alert, Platform,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import * as Speech from 'expo-speech';
import * as Clipboard from 'expo-clipboard';
import axios from 'axios';
import { API_URL } from '../config';
import { useTheme } from '../theme';
import { useI18n } from '../i18n';
import { saveToHistory } from '../storage';

const STATIC_MS     = 2000;  // intervalo modo continuo estático (30 req/min = 1 cada 2s)
const DYNAMIC_MS    = 200;   // intervalo de captura de frames dinámicos (~5 fps)
const FRAMES_NEEDED = 30;    // debe coincidir con SEQUENCE_LENGTH del backend

/**
 * Construye un FormData compatible con web y nativo.
 * En web, expo-camera devuelve un blob:// URL que hay que
 * convertir a Blob real antes de añadirlo al FormData.
 */
async function buildFormData(photoUri) {
  const fd = new FormData();
  if (Platform.OS === 'web') {
    const res  = await fetch(photoUri);
    const blob = await res.blob();
    fd.append('file', blob, 'photo.jpg');
  } else {
    fd.append('file', { uri: photoUri, name: 'photo.jpg', type: 'image/jpeg' });
  }
  return fd;
}

export default function SignToTextScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef(null);

  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence]  = useState(0);
  const [loading, setLoading]        = useState(false);
  const [copied, setCopied]          = useState(false);

  // Modos: 'static' | 'dynamic'
  const [mode, setMode]              = useState('static');
  // Modo continuo (solo aplica a static)
  const [continuousMode, setContinuousMode] = useState(false);
  // Progreso de captura dinámica (0-30)
  const [frameCount, setFrameCount]  = useState(0);

  const intervalRef = useRef(null);

  const fadeAnim  = useRef(new Animated.Value(0)).current;
  const scaleAnim = useRef(new Animated.Value(0.85)).current;

  const theme = useTheme();
  const t     = useI18n().signToText;
  const s     = styles(theme);

  // Limpia el intervalo al cambiar de modo
  useEffect(() => {
    clearInterval(intervalRef.current);
    setFrameCount(0);
    setPrediction('');
    setConfidence(0);

    if (mode === 'static' && continuousMode) {
      intervalRef.current = setInterval(captureStatic, STATIC_MS);
    } else if (mode === 'dynamic') {
      _resetSequence();
      intervalRef.current = setInterval(captureDynamicFrame, DYNAMIC_MS);
    }

    return () => clearInterval(intervalRef.current);
  }, [mode, continuousMode]);

  const animateResult = useCallback(() => {
    fadeAnim.setValue(0);
    scaleAnim.setValue(0.85);
    Animated.parallel([
      Animated.timing(fadeAnim,  { toValue: 1, duration: 250, useNativeDriver: true }),
      Animated.spring(scaleAnim, { toValue: 1, friction: 6,   useNativeDriver: true }),
    ]).start();
  }, [fadeAnim, scaleAnim]);

  if (!permission) return <View style={s.container} />;

  if (!permission.granted) {
    return (
      <View style={[s.container, s.center]}>
        <Text style={s.permText}>{t.cameraNeeded}</Text>
        <TouchableOpacity
          style={s.permBtn}
          onPress={requestPermission}
          accessibilityRole="button"
          accessibilityLabel={t.allowCamera}
        >
          <Text style={s.permBtnText}>{t.allowCamera}</Text>
        </TouchableOpacity>
      </View>
    );
  }

  // ── Detección estática ─────────────────────────────────────────────────────

  const captureStatic = async () => {
    if (!cameraRef.current || loading) return;
    setLoading(true);
    try {
      const photo = await cameraRef.current.takePictureAsync({ base64: false });
      const formData = await buildFormData(photo.uri);

      const { data } = await axios.post(`${API_URL}/predict-sign`, formData, {
        timeout: 8000,
      });

      const pred = data.prediction ?? '—';
      const conf = data.confidence ?? 0;
      setPrediction(pred);
      setConfidence(conf);
      animateResult();

      if (pred !== 'No detectado' && pred !== 'No reconocido') {
        saveToHistory({ type: 'sign_to_text', input: '[imagen]', output: pred, confidence: conf });
      }
    } catch (err) {
      const status = err.response?.status;
      const msg =
        err.code === 'ECONNABORTED' ? 'Tiempo de espera agotado. ¿Está el servidor activo?' :
        !err.response              ? t.connectionError :
        status === 429             ? 'Demasiadas peticiones. Espera un momento.' :
        status === 422             ? 'Imagen no válida. Intenta de nuevo.' :
        `Error del servidor (${status})`;
      if (!continuousMode) Alert.alert('Error', msg);
      else { setPrediction(msg); setConfidence(0); }
    } finally {
      setLoading(false);
    }
  };

  // ── Detección dinámica ─────────────────────────────────────────────────────

  const _resetSequence = async () => {
    try { await axios.delete(`${API_URL}/predict-sign-sequence/reset`, { timeout: 3000 }); }
    catch (_) {}
    setFrameCount(0);
  };

  const captureDynamicFrame = async () => {
    if (!cameraRef.current) return;
    try {
      const photo = await cameraRef.current.takePictureAsync({ base64: false, quality: 0.5 });
      const formData = await buildFormData(photo.uri);

      const { data } = await axios.post(`${API_URL}/predict-sign-sequence/frame`, formData, {
        timeout: 4000,
      });

      setFrameCount(prev => {
        const next = Math.min(prev + 1, FRAMES_NEEDED);
        return next;
      });

      if (data.ready) {
        clearInterval(intervalRef.current);
        await _predictDynamic();
        // Reinicia captura para la siguiente seña
        await _resetSequence();
        intervalRef.current = setInterval(captureDynamicFrame, DYNAMIC_MS);
      }
    } catch (_) {}
  };

  const _predictDynamic = async () => {
    setLoading(true);
    try {
      const { data } = await axios.get(`${API_URL}/predict-sign-sequence/predict`, { timeout: 6000 });
      const pred = data.prediction ?? '—';
      const conf = data.confidence ?? 0;

      if (pred !== 'Capturando...' && pred !== 'Modelo no disponible') {
        setPrediction(pred);
        setConfidence(conf);
        animateResult();
        if (pred !== 'No reconocido') {
          saveToHistory({ type: 'sign_to_text_dynamic', input: '[secuencia]', output: pred, confidence: conf });
        }
      }
    } catch (_) {}
    finally { setLoading(false); }
  };

  // ── Acciones UI ────────────────────────────────────────────────────────────

  const handleSpeak = () => {
    if (prediction) Speech.speak(prediction, { language: 'es-MX' });
  };

  const handleCopy = async () => {
    if (!prediction) return;
    await Clipboard.setStringAsync(prediction);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  const toggleMode = (newMode) => {
    setContinuousMode(false);
    setMode(newMode);
  };

  const confColor =
    confidence >= 0.8 ? theme.success :
    confidence >= 0.6 ? theme.warning : theme.error;

  const dynamicProgress = Math.round((frameCount / FRAMES_NEEDED) * 100);

  return (
    <View style={s.container}>
      <CameraView
        style={s.camera}
        facing="front"
        ref={cameraRef}
        accessibilityLabel="Vista de cámara frontal"
      />

      <View style={s.panel}>

        {/* Selector de modo */}
        <View style={s.modeRow}>
          <TouchableOpacity
            style={[s.modeBtn, mode === 'static'  && s.modeBtnActive]}
            onPress={() => toggleMode('static')}
            accessibilityRole="button"
            accessibilityLabel="Modo estático"
          >
            <Text style={[s.modeBtnText, mode === 'static' && s.modeBtnTextActive]}>
              Estático
            </Text>
          </TouchableOpacity>
          <TouchableOpacity
            style={[s.modeBtn, mode === 'dynamic' && s.modeBtnActive]}
            onPress={() => toggleMode('dynamic')}
            accessibilityRole="button"
            accessibilityLabel="Modo dinámico"
          >
            <Text style={[s.modeBtnText, mode === 'dynamic' && s.modeBtnTextActive]}>
              Dinámico
            </Text>
          </TouchableOpacity>
        </View>

        {/* Controles modo estático */}
        {mode === 'static' && (
          <>
            <View style={s.row}>
              <Text style={s.label}>{t.continuousMode}</Text>
              <Switch
                value={continuousMode}
                onValueChange={setContinuousMode}
                trackColor={{ true: theme.primary }}
                accessibilityRole="switch"
                accessibilityLabel={t.continuousMode}
              />
            </View>
            {!continuousMode && (
              <TouchableOpacity
                style={s.detectBtn}
                onPress={captureStatic}
                disabled={loading}
                accessibilityRole="button"
                accessibilityLabel={t.detectBtn}
              >
                <Text style={s.detectBtnText}>{t.detectBtn}</Text>
              </TouchableOpacity>
            )}
          </>
        )}

        {/* Progreso modo dinámico */}
        {mode === 'dynamic' && (
          <View>
            <Text style={[s.label, { marginBottom: 4 }]}>
              Capturando seña… {frameCount}/{FRAMES_NEEDED} frames
            </Text>
            <View style={s.progressBar}>
              <View style={[s.progressFill, { width: `${dynamicProgress}%` }]} />
            </View>
          </View>
        )}

        {loading && (
          <ActivityIndicator color={theme.primary} style={{ marginTop: 8 }} />
        )}

        {prediction !== '' && (
          <Animated.View
            style={[s.resultBox, { opacity: fadeAnim, transform: [{ scale: scaleAnim }] }]}
            accessibilityLiveRegion="polite"
          >
            <Text style={s.predictionText} accessibilityRole="text">
              {prediction}
            </Text>
            {confidence > 0 && (
              <Text style={[s.confidenceText, { color: confColor }]}>
                {t.confidence}: {(confidence * 100).toFixed(0)}%
              </Text>
            )}
            <View style={s.actions}>
              <TouchableOpacity
                style={[s.actionBtn, { backgroundColor: theme.primary }]}
                onPress={handleSpeak}
                accessibilityRole="button"
                accessibilityLabel={t.listen}
              >
                <Text style={s.actionBtnText}>{t.listen}</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[s.actionBtn, { backgroundColor: theme.surface, borderWidth: 1, borderColor: theme.border }]}
                onPress={handleCopy}
                accessibilityRole="button"
                accessibilityLabel={copied ? t.copied : t.copy}
              >
                <Text style={[s.actionBtnText, { color: theme.text }]}>
                  {copied ? t.copied : t.copy}
                </Text>
              </TouchableOpacity>
            </View>
          </Animated.View>
        )}
      </View>
    </View>
  );
}

const styles = (t) => StyleSheet.create({
  container:       { flex: 1, backgroundColor: t.background },
  center:          { justifyContent: 'center', alignItems: 'center', padding: 24, gap: 16 },
  camera:          { flex: 1 },
  panel:           {
    position: 'absolute', bottom: 0, left: 0, right: 0,
    backgroundColor: t.card,
    padding: 16, gap: 10,
    borderTopLeftRadius: 16, borderTopRightRadius: 16,
    elevation: 8,
    boxShadow: '0px -2px 8px rgba(0,0,0,0.10)',
  },
  row:             { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  label:           { fontSize: 14, color: t.text },
  modeRow:         { flexDirection: 'row', borderRadius: 8, overflow: 'hidden', borderWidth: 1, borderColor: t.border },
  modeBtn:         { flex: 1, padding: 8, alignItems: 'center', backgroundColor: t.surface },
  modeBtnActive:   { backgroundColor: t.primary },
  modeBtnText:     { fontSize: 14, fontWeight: '600', color: t.text },
  modeBtnTextActive: { color: '#FFF' },
  detectBtn:       { backgroundColor: t.primary, borderRadius: 10, padding: 14, alignItems: 'center' },
  detectBtnText:   { color: '#FFF', fontSize: 16, fontWeight: '700' },
  progressBar:     { height: 8, borderRadius: 4, backgroundColor: t.border, overflow: 'hidden' },
  progressFill:    { height: '100%', backgroundColor: t.primary, borderRadius: 4 },
  resultBox:       { alignItems: 'center', paddingTop: 8, gap: 6 },
  predictionText:  { fontSize: 44, fontWeight: '800', color: t.text, textAlign: 'center' },
  confidenceText:  { fontSize: 14 },
  actions:         { flexDirection: 'row', gap: 10, marginTop: 4 },
  actionBtn:       { flex: 1, borderRadius: 8, padding: 10, alignItems: 'center' },
  actionBtnText:   { fontSize: 14, fontWeight: '600', color: '#FFF' },
  permText:        { fontSize: 16, color: t.text, textAlign: 'center' },
  permBtn:         { backgroundColor: t.primary, borderRadius: 10, padding: 14, paddingHorizontal: 24 },
  permBtnText:     { color: '#FFF', fontWeight: '700', fontSize: 15 },
});
