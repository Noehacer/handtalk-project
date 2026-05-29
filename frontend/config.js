import Constants from 'expo-constants';

// La URL se lee desde app.json → extra → apiUrl
// Cámbiala ahí y no tendrás que tocar código fuente.
export const API_URL =
  Constants.expoConfig?.extra?.apiUrl ?? 'http://localhost:8000';
