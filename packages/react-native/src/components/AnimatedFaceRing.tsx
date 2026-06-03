/**
 * AnimatedFaceRing — pulsing circular ring used around the face detection
 * area. Built on RN's built-in Animated API (no Reanimated dependency).
 *
 * Props:
 *   - size: outer ring diameter
 *   - active: when true, the ring breathes + the progress arc rotates
 *   - progress: 0..1 sweep (used to indicate enrollment progress)
 *   - color: ring colour
 */

import React, { useEffect, useRef } from 'react';
import { Animated, Easing, StyleSheet, View } from 'react-native';
import Svg, { Circle } from 'react-native-svg';

const AnimatedCircle = Animated.createAnimatedComponent(Circle);

export interface AnimatedFaceRingProps {
  size: number;
  active: boolean;
  progress?: number; // 0..1
  color?: string;
  trackColor?: string;
}

export function AnimatedFaceRing({
  size,
  active,
  progress = 0,
  color = '#3b82f6',
  trackColor = 'rgba(255,255,255,0.08)',
}: AnimatedFaceRingProps): React.JSX.Element {
  const breathe = useRef(new Animated.Value(0)).current;
  const sweep = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (!active) {
      breathe.stopAnimation();
      sweep.stopAnimation();
      return;
    }
    const breatheLoop = Animated.loop(
      Animated.sequence([
        Animated.timing(breathe, { toValue: 1, duration: 1200, easing: Easing.inOut(Easing.quad), useNativeDriver: true }),
        Animated.timing(breathe, { toValue: 0, duration: 1200, easing: Easing.inOut(Easing.quad), useNativeDriver: true }),
      ]),
    );
    const sweepLoop = Animated.loop(
      Animated.timing(sweep, { toValue: 1, duration: 1800, easing: Easing.linear, useNativeDriver: true }),
    );
    breatheLoop.start();
    sweepLoop.start();
    return () => { breatheLoop.stop(); sweepLoop.stop(); };
  }, [active, breathe, sweep]);

  const r = (size - 12) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const circumference = 2 * Math.PI * r;
  const dashOffset = circumference * (1 - Math.min(1, Math.max(0, progress)));
  const sweepDeg = sweep.interpolate({ inputRange: [0, 1], outputRange: ['0deg', '360deg'] });
  const scale = breathe.interpolate({ inputRange: [0, 1], outputRange: [1, 1.04] });
  const opacity = breathe.interpolate({ inputRange: [0, 1], outputRange: [0.85, 1] });

  return (
    <View style={{ width: size, height: size }}>
      <Animated.View
        style={[
          StyleSheet.absoluteFillObject,
          { alignItems: 'center', justifyContent: 'center', transform: [{ scale }], opacity },
        ]}
      >
        <Svg width={size} height={size}>
          <Circle cx={cx} cy={cy} r={r} stroke={trackColor} strokeWidth={3} fill="none" />
          <AnimatedCircle
            cx={cx}
            cy={cy}
            r={r}
            stroke={color}
            strokeWidth={3}
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            strokeLinecap="round"
            transform={`rotate(-90 ${cx} ${cy})`}
          />
        </Svg>
      </Animated.View>
      {active && (
        <Animated.View
          pointerEvents="none"
          style={[
            StyleSheet.absoluteFillObject,
            { alignItems: 'center', justifyContent: 'center', transform: [{ rotate: sweepDeg }] },
          ]}
        >
          <View
            style={{
              position: 'absolute',
              top: 4,
              width: 8,
              height: 8,
              borderRadius: 4,
              backgroundColor: color,
              shadowColor: color,
              shadowOpacity: 0.9,
              shadowRadius: 8,
              shadowOffset: { width: 0, height: 0 },
            }}
          />
        </Animated.View>
      )}
    </View>
  );
}
