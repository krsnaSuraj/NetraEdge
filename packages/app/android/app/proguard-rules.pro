# NetraEdge ProGuard Rules

# Keep TFLite classes
-keep class org.tensorflow.lite.** { *; }
-keepclassmembers class org.tensorflow.lite.** { *; }

# Keep NetraEdge native modules
-keep class com.netraedge.NetraEdgeModule { *; }
-keep class com.netraedge.NetraEdgePackage { *; }
-keep class com.netraedge.NetraEdgeNativeModule { *; }
-keep class com.netraedge.FaceCropPlugin { *; }

# Keep React Native bridge classes
-keep class com.facebook.react.bridge.** { *; }
-keep class com.facebook.react.modules.** { *; }

# Don't obfuscate native module names (React Native needs them)
-keepnames class * extends com.facebook.react.bridge.NativeModule
-keepnames class * extends com.facebook.react.uimanager.ViewManager

# Keep enum values
-keepclassmembers enum * {
    public static **[] values();
    public static ** valueOf(java.lang.String);
}

# Keep Parcelable
-keepclassmembers class * implements android.os.Parcelable {
    public static final ** CREATOR;
}

# Suppress warnings for missing annotations
-dontwarn org.tensorflow.lite.**
-dontwarn javax.annotation.**
