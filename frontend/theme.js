import { useColorScheme } from 'react-native';

const light = {
  background:     '#FFFFFF',
  surface:        '#F2F2F7',
  card:           '#FFFFFF',
  text:           '#1A1A1A',
  textSecondary:  '#6C6C70',
  border:         '#E0E0E0',
  primary:        '#007AFF',
  success:        '#34C759',
  warning:        '#FF9500',
  error:          '#FF3B30',
};

const dark = {
  background:     '#000000',
  surface:        '#1C1C1E',
  card:           '#2C2C2E',
  text:           '#F2F2F7',
  textSecondary:  '#8E8E93',
  border:         '#38383A',
  primary:        '#0A84FF',
  success:        '#30D158',
  warning:        '#FF9F0A',
  error:          '#FF453A',
};

export function useTheme() {
  const scheme = useColorScheme();
  return scheme === 'dark' ? dark : light;
}
