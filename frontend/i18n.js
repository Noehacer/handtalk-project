import { NativeModules, Platform } from 'react-native';

const STRINGS = {
  es: {
    home: {
      subtitle:        'Traductor de Lengua de Señas Mexicana',
      signsToText:     'Señas → Texto',
      textToSigns:     'Texto → Señas',
      history:         'Historial',
      signsToTextSub:  'Usa la cámara para detectar señas',
      textToSignsSub:  'Escribe una palabra o frase',
      historySub:      'Ver traducciones anteriores',
    },
    signToText: {
      continuousMode:  'Modo continuo',
      detectBtn:       'Detectar seña',
      listen:          '🔊 Escuchar',
      copy:            '📋 Copiar',
      copied:          '✓ Copiado',
      connectionError: 'Sin conexión al servidor',
      allowCamera:     'Permitir cámara',
      cameraNeeded:    'Se necesita acceso a la cámara',
      confidence:      'Confianza',
    },
    textToSign: {
      title:           'Texto → Señas',
      placeholder:     'Escribe una palabra o frase',
      translate:       'Traducir',
      translatePhrase: 'Traducir frase',
      notFound:        'Seña no encontrada',
      imgUnavailable:  'Imagen no disponible',
      didYouMean:      '¿Quisiste decir:',
      connectionError: 'Sin conexión al servidor',
      words:           'palabras',
    },
    history: {
      empty:           'Sin traducciones aún',
      emptySub:        'Las traducciones aparecerán aquí',
      clearHistory:    'Borrar historial',
      clearConfirm:    '¿Deseas eliminar todas las traducciones guardadas?',
      cancel:          'Cancelar',
      delete:          'Borrar',
      confidence:      'Confianza',
    },
    onboarding: {
      skip:  'Omitir',
      next:  'Siguiente',
      start: 'Comenzar',
      slides: [
        {
          emoji:       '🤟',
          title:       'Bienvenido a HandTalk',
          description: 'Tu traductor de Lengua de Señas Mexicana. Aprende y comunícate con señas.',
        },
        {
          emoji:       '📷',
          title:       'Señas → Texto',
          description: 'Muestra una seña frente a la cámara y la app la reconocerá automáticamente.',
        },
        {
          emoji:       '⌨️',
          title:       'Texto → Señas',
          description: 'Escribe cualquier palabra o frase y verás cómo se expresa en lengua de señas.',
        },
      ],
    },
  },

  en: {
    home: {
      subtitle:        'Mexican Sign Language Translator',
      signsToText:     'Signs → Text',
      textToSigns:     'Text → Signs',
      history:         'History',
      signsToTextSub:  'Use the camera to detect signs',
      textToSignsSub:  'Type a word or phrase',
      historySub:      'View past translations',
    },
    signToText: {
      continuousMode:  'Continuous mode',
      detectBtn:       'Detect sign',
      listen:          '🔊 Listen',
      copy:            '📋 Copy',
      copied:          '✓ Copied',
      connectionError: 'No server connection',
      allowCamera:     'Allow camera',
      cameraNeeded:    'Camera access needed',
      confidence:      'Confidence',
    },
    textToSign: {
      title:           'Text → Signs',
      placeholder:     'Type a word or phrase',
      translate:       'Translate',
      translatePhrase: 'Translate phrase',
      notFound:        'Sign not found',
      imgUnavailable:  'Image unavailable',
      didYouMean:      'Did you mean:',
      connectionError: 'No server connection',
      words:           'words',
    },
    history: {
      empty:           'No translations yet',
      emptySub:        'Your translations will appear here',
      clearHistory:    'Clear history',
      clearConfirm:    'Delete all saved translations?',
      cancel:          'Cancel',
      delete:          'Delete',
      confidence:      'Confidence',
    },
    onboarding: {
      skip:  'Skip',
      next:  'Next',
      start: 'Get Started',
      slides: [
        {
          emoji:       '🤟',
          title:       'Welcome to HandTalk',
          description: 'Your Mexican Sign Language translator. Learn and communicate with signs.',
        },
        {
          emoji:       '📷',
          title:       'Signs → Text',
          description: 'Show a sign in front of the camera and the app will recognize it automatically.',
        },
        {
          emoji:       '⌨️',
          title:       'Text → Signs',
          description: 'Type any word or phrase and see how it is expressed in sign language.',
        },
      ],
    },
  },
};

function detectLanguage() {
  try {
    const locale = Platform.OS === 'ios'
      ? (NativeModules.SettingsManager?.settings?.AppleLocale ||
         NativeModules.SettingsManager?.settings?.AppleLanguages?.[0] || 'es')
      : (NativeModules.I18nManager?.localeIdentifier || 'es');
    return locale.startsWith('en') ? 'en' : 'es';
  } catch {
    return 'es';
  }
}

let _lang = detectLanguage();

export function useI18n() {
  return STRINGS[_lang] ?? STRINGS.es;
}

export function setLanguage(lang) {
  if (STRINGS[lang]) _lang = lang;
}

export function getLanguage() {
  return _lang;
}

export const AVAILABLE_LANGUAGES = [
  { code: 'es', label: 'Español' },
  { code: 'en', label: 'English' },
];
