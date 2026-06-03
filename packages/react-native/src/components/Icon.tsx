/**
 * Icon — vector icon set rendered with react-native-svg paths.
 *
 * Inline SVG path strings keep the bundle lean (no icon font
 * or external SVG library required at this level). To upgrade
 * to react-native-vector-icons, replace the inner render with
 * <Icon name={name} ... /> from that library.
 *
 * All icons are drawn on a 24x24 viewBox and inherit currentColor.
 */

import React from 'react';
import { View } from 'react-native';
import Svg, { Path, Circle, Rect, Line, Polyline } from 'react-native-svg';

export type IconName =
  | 'face' | 'shield' | 'cloud' | 'check' | 'x' | 'arrow-left' | 'arrow-right'
  | 'settings' | 'plus' | 'refresh' | 'eye' | 'eye-off' | 'user'
  | 'trash' | 'download' | 'upload' | 'globe' | 'lock' | 'unlock'
  | 'camera' | 'scan' | 'alert' | 'info' | 'bolt' | 'sun' | 'moon' | 'wifi';

export interface IconProps {
  name: IconName;
  size?: number;
  color?: string;
  strokeWidth?: number;
}

export function Icon({
  name, size = 24, color = '#ffffff', strokeWidth = 1.6,
}: IconProps): React.JSX.Element {
  return (
    <View style={{ width: size, height: size }}>
      <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
        {renderPath(name, color, strokeWidth)}
      </Svg>
    </View>
  );
}

function renderPath(name: IconName, color: string, sw: number): React.ReactNode {
  const p = (d: string) => <Path d={d} stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" />;
  switch (name) {
    case 'face':
      return (
        <>
          <Circle cx="12" cy="12" r="9" stroke={color} strokeWidth={sw} />
          <Circle cx="9" cy="10" r="1" fill={color} />
          <Circle cx="15" cy="10" r="1" fill={color} />
          <Path d="M8.5 15c1 1.2 2.2 1.8 3.5 1.8s2.5-.6 3.5-1.8" stroke={color} strokeWidth={sw} strokeLinecap="round" />
        </>
      );
    case 'shield':
      return p('M12 2 4 5v6c0 5 3.5 9 8 11 4.5-2 8-6 8-11V5l-8-3Z');
    case 'cloud':
      return (
        <>
          <Path d="M7 18a4 4 0 0 1-.5-7.97 6 6 0 0 1 11.5 2c2 .5 3 2.5 2.5 4.5S18 19 16 19H7Z" stroke={color} strokeWidth={sw} strokeLinejoin="round" />
        </>
      );
    case 'check':
      return <Polyline points="4,12 10,18 20,6" stroke={color} strokeWidth={sw + 0.4} strokeLinecap="round" strokeLinejoin="round" fill="none" />;
    case 'x':
      return (
        <>
          <Line x1="6" y1="6" x2="18" y2="18" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Line x1="18" y1="6" x2="6" y2="18" stroke={color} strokeWidth={sw} strokeLinecap="round" />
        </>
      );
    case 'arrow-left':
      return (
        <>
          <Line x1="20" y1="12" x2="4" y2="12" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Polyline points="10,6 4,12 10,18" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" fill="none" />
        </>
      );
    case 'arrow-right':
      return (
        <>
          <Line x1="4" y1="12" x2="20" y2="12" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Polyline points="14,6 20,12 14,18" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" fill="none" />
        </>
      );
    case 'settings':
      return (
        <>
          <Circle cx="12" cy="12" r="3" stroke={color} strokeWidth={sw} />
          <Path d="M19.4 15a1.7 1.7 0 0 0 .34 1.85l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.85-.34 1.7 1.7 0 0 0-1.05 1.55V21a2 2 0 1 1-4 0v-.09a1.7 1.7 0 0 0-1.11-1.55 1.7 1.7 0 0 0-1.85.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-1.55-1.05H3a2 2 0 1 1 0-4h.09A1.7 1.7 0 0 0 4.6 8.85a1.7 1.7 0 0 0-.34-1.85l-.06-.06A2 2 0 1 1 7.03 4.1l.06.06a1.7 1.7 0 0 0 1.85.34H9a1.7 1.7 0 0 0 1.05-1.55V3a2 2 0 1 1 4 0v.09A1.7 1.7 0 0 0 15 4.6a1.7 1.7 0 0 0 1.85-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.34 1.85V9a1.7 1.7 0 0 0 1.55 1.05H21a2 2 0 1 1 0 4h-.09a1.7 1.7 0 0 0-1.55 1Z" stroke={color} strokeWidth={sw} />
        </>
      );
    case 'plus':
      return (
        <>
          <Line x1="12" y1="5" x2="12" y2="19" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Line x1="5" y1="12" x2="19" y2="12" stroke={color} strokeWidth={sw} strokeLinecap="round" />
        </>
      );
    case 'refresh':
      return (
        <>
          <Polyline points="3,12 6,9 9,12" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <Path d="M6 9v4a6 6 0 0 0 11 2" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Polyline points="21,12 18,15 15,12" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <Path d="M18 15v-4a6 6 0 0 0-11-2" stroke={color} strokeWidth={sw} strokeLinecap="round" />
        </>
      );
    case 'eye':
      return (
        <>
          <Path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12Z" stroke={color} strokeWidth={sw} />
          <Circle cx="12" cy="12" r="3" stroke={color} strokeWidth={sw} />
        </>
      );
    case 'eye-off':
      return (
        <>
          <Path d="M3 3l18 18" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Path d="M10.5 6.2A10.9 10.9 0 0 1 12 6c6.5 0 10 6 10 6a14 14 0 0 1-3 3.7" stroke={color} strokeWidth={sw} />
          <Path d="M6.6 6.6A14 14 0 0 0 2 12s3.5 6 10 6a10.7 10.7 0 0 0 4.4-.9" stroke={color} strokeWidth={sw} />
        </>
      );
    case 'user':
      return (
        <>
          <Circle cx="12" cy="8" r="4" stroke={color} strokeWidth={sw} />
          <Path d="M4 21a8 8 0 0 1 16 0" stroke={color} strokeWidth={sw} strokeLinecap="round" />
        </>
      );
    case 'trash':
      return (
        <>
          <Polyline points="3,6 5,6 21,6" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <Path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" />
        </>
      );
    case 'download':
      return (
        <>
          <Line x1="12" y1="3" x2="12" y2="15" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Polyline points="7,10 12,15 17,10" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <Line x1="3" y1="21" x2="21" y2="21" stroke={color} strokeWidth={sw} strokeLinecap="round" />
        </>
      );
    case 'upload':
      return (
        <>
          <Line x1="12" y1="21" x2="12" y2="9" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Polyline points="7,14 12,9 17,14" stroke={color} strokeWidth={sw} strokeLinecap="round" strokeLinejoin="round" fill="none" />
          <Line x1="3" y1="3" x2="21" y2="3" stroke={color} strokeWidth={sw} strokeLinecap="round" />
        </>
      );
    case 'globe':
      return (
        <>
          <Circle cx="12" cy="12" r="9" stroke={color} strokeWidth={sw} />
          <Path d="M3 12h18M12 3a14 14 0 0 1 0 18M12 3a14 14 0 0 0 0 18" stroke={color} strokeWidth={sw} />
        </>
      );
    case 'lock':
      return (
        <>
          <Rect x="4" y="11" width="16" height="10" rx="2" stroke={color} strokeWidth={sw} />
          <Path d="M8 11V7a4 4 0 0 1 8 0v4" stroke={color} strokeWidth={sw} />
        </>
      );
    case 'unlock':
      return (
        <>
          <Rect x="4" y="11" width="16" height="10" rx="2" stroke={color} strokeWidth={sw} />
          <Path d="M8 11V7a4 4 0 0 1 7-2.5" stroke={color} strokeWidth={sw} />
        </>
      );
    case 'camera':
      return (
        <>
          <Path d="M3 7h4l2-3h6l2 3h4v13H3V7Z" stroke={color} strokeWidth={sw} strokeLinejoin="round" />
          <Circle cx="12" cy="13" r="4" stroke={color} strokeWidth={sw} />
        </>
      );
    case 'scan':
      return (
        <>
          <Path d="M3 7V5a2 2 0 0 1 2-2h2M21 7V5a2 2 0 0 0-2-2h-2M3 17v2a2 2 0 0 0 2 2h2M21 17v2a2 2 0 0 1-2 2h-2" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Line x1="7" y1="12" x2="17" y2="12" stroke={color} strokeWidth={sw} strokeLinecap="round" />
        </>
      );
    case 'alert':
      return (
        <>
          <Path d="M12 3 2 21h20L12 3Z" stroke={color} strokeWidth={sw} strokeLinejoin="round" />
          <Line x1="12" y1="10" x2="12" y2="14" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Circle cx="12" cy="17.5" r="0.8" fill={color} />
        </>
      );
    case 'info':
      return (
        <>
          <Circle cx="12" cy="12" r="9" stroke={color} strokeWidth={sw} />
          <Line x1="12" y1="11" x2="12" y2="16" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Circle cx="12" cy="8" r="0.8" fill={color} />
        </>
      );
    case 'bolt':
      return p('M13 2 4 14h7l-1 8 9-12h-7l1-8Z');
    case 'sun':
      return (
        <>
          <Circle cx="12" cy="12" r="4" stroke={color} strokeWidth={sw} />
          <Line x1="12" y1="2" x2="12" y2="5" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Line x1="12" y1="19" x2="12" y2="22" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Line x1="2" y1="12" x2="5" y2="12" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Line x1="19" y1="12" x2="22" y2="12" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Line x1="4.5" y1="4.5" x2="6.5" y2="6.5" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Line x1="17.5" y1="17.5" x2="19.5" y2="19.5" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Line x1="4.5" y1="19.5" x2="6.5" y2="17.5" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Line x1="17.5" y1="6.5" x2="19.5" y2="4.5" stroke={color} strokeWidth={sw} strokeLinecap="round" />
        </>
      );
    case 'moon':
      return p('M21 13a9 9 0 0 1-10-10 9 9 0 1 0 10 10Z');
    case 'wifi':
      return (
        <>
          <Path d="M5 12.5a10 10 0 0 1 14 0" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Path d="M8.5 16a5 5 0 0 1 7 0" stroke={color} strokeWidth={sw} strokeLinecap="round" />
          <Circle cx="12" cy="20" r="1" fill={color} />
        </>
      );
  }
}
