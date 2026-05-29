import React, { useState, useEffect, useCallback } from 'react';
import { useColorScheme, View } from 'react-native';
import { NavigationContainer, DarkTheme, DefaultTheme } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import * as SplashScreen from 'expo-splash-screen';

import HomeScreen from './screens/HomeScreen';
import SignToTextScreen from './screens/SignToTextScreen';
import TextToSignScreen from './screens/TextToSignScreen';
import HistoryScreen from './screens/HistoryScreen';
import OnboardingScreen, { isOnboardingSeen } from './screens/OnboardingScreen';

// Mantener el splash nativo visible mientras se inicializa la app
SplashScreen.preventAutoHideAsync();

const Stack = createNativeStackNavigator();

export default function App() {
  const scheme = useColorScheme();
  const [appReady, setAppReady] = useState(false);
  const [showOnboarding, setShowOnboarding] = useState(false);

  useEffect(() => {
    async function prepare() {
      try {
        const seen = await isOnboardingSeen();
        setShowOnboarding(!seen);
      } catch {
        setShowOnboarding(false);
      } finally {
        setAppReady(true);
        await SplashScreen.hideAsync();
      }
    }
    prepare();
  }, []);

  if (!appReady) return <View />;

  if (showOnboarding) {
    return <OnboardingScreen onDone={() => setShowOnboarding(false)} />;
  }

  return (
    <NavigationContainer theme={scheme === 'dark' ? DarkTheme : DefaultTheme}>
      <Stack.Navigator
        screenOptions={{
          animation: 'slide_from_right',
          headerTitleStyle: { fontWeight: '700' },
        }}
      >
        <Stack.Screen
          name="Home"
          component={HomeScreen}
          options={{ title: 'HandTalk', headerShown: false }}
        />
        <Stack.Screen
          name="SignToText"
          component={SignToTextScreen}
          options={{ title: 'Señas → Texto' }}
        />
        <Stack.Screen
          name="TextToSign"
          component={TextToSignScreen}
          options={{ title: 'Texto → Señas' }}
        />
        <Stack.Screen
          name="History"
          component={HistoryScreen}
          options={{ title: 'Historial' }}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}
