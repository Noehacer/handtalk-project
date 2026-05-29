import AsyncStorage from '@react-native-async-storage/async-storage';

const KEY = 'handtalk_history';
const MAX_ENTRIES = 50;

export async function saveToHistory(entry) {
  try {
    const raw = await AsyncStorage.getItem(KEY);
    const history = raw ? JSON.parse(raw) : [];
    history.unshift({ ...entry, date: new Date().toISOString() });
    await AsyncStorage.setItem(KEY, JSON.stringify(history.slice(0, MAX_ENTRIES)));
  } catch {
    // Silencioso: el historial no es crítico
  }
}

export async function loadHistory() {
  try {
    const raw = await AsyncStorage.getItem(KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

export async function clearHistory() {
  await AsyncStorage.removeItem(KEY);
}
