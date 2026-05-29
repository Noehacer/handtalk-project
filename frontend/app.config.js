export default ({ config }) => ({
  name: 'HandTalk',
  slug: 'handtalk',
  version: '1.0.0',
  orientation: 'portrait',
  userInterfaceStyle: 'automatic',
  icon: './assets/icon.png',
  splash: {
    image: './assets/splash.png',
    resizeMode: 'contain',
    backgroundColor: '#007AFF',
  },
  ios: {
    supportsTablet: false,
    bundleIdentifier: 'com.handtalk.app',
    infoPlist: {
      NSCameraUsageDescription: 'HandTalk necesita la cámara para detectar señas.',
      NSSpeechRecognitionUsageDescription: 'HandTalk usa síntesis de voz para leer traducciones.',
    },
  },
  android: {
    package: 'com.handtalk.app',
    adaptiveIcon: {
      foregroundImage: './assets/adaptive-icon.png',
      backgroundColor: '#007AFF',
    },
    permissions: [
      'android.permission.CAMERA',
      'android.permission.RECORD_AUDIO',
    ],
  },
  plugins: [
    [
      'expo-camera',
      { cameraPermission: 'HandTalk necesita acceso a la cámara para detectar señas.' },
    ],
  ],
  updates: {
    enabled: false,
  },
  extra: {
    apiUrl: process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8000',
  },
});
