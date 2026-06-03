# NetraEdge ProGuard Rules

# Keep TFLite classes
-keep class org.tensorflow.lite.** { *; }
-keepclassmembers class org.tensorflow.lite.** { *; }

# Keep MediaPipe Tasks Vision classes
-keep class com.google.mediapipe.** { *; }
-keepclassmembers class com.google.mediapipe.** { *; }
-dontwarn com.google.mediapipe.**

# Keep NetraEdge native modules
-keep class com.netraedge.NetraEdgeModule { *; }
-keep class com.netraedge.NetraEdgePackage { *; }
-keep class com.netraedge.NetraEdgeNativeModule { *; }
-keep class com.netraedge.FaceCropPlugin { *; }
-keep class com.netraedge.MainActivity { *; }
-keep class com.netraedge.FaceOverlayView { *; }
-keep class com.netraedge.FaceAligner { *; }
-keep class com.netraedge.KeypointExtractor { *; }
-keep class com.netraedge.RppgAnalyzer { *; }
-keep class com.netraedge.ActiveChallengeRunner { *; }
-keep class com.netraedge.FFT { *; }

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
-dontwarn com.google.protobuf.**
-dontwarn autovalue.shaded.com.squareup.javapoet.**
-dontwarn javax.lang.model.**
-dontwarn com.google.auto.value.**
-dontwarn com.google.auto.**
-dontwarn org.checkerframework.**
-dontwarn org.codehaus.mojo.animal_sniffer.**
-dontwarn com.google.errorprone.**
-dontwarn com.google.common.**
-dontwarn com.google.j2objc.**
-dontwarn org.conscrypt.**
-dontwarn org.openjsse.**
-dontwarn org.bouncycastle.**

# Aggressive obfuscation for non-bridge classes
-allowaccessmodification
-repackageclasses 'n'
-flattenpackagehierarchy ''
-overloadaggressively

# Keep line numbers (for crash reports) but rename source files
-keepattributes SourceFile,LineNumberTable
-renamesourcefileattribute SourceFile

# Keep critical classes only
-keep class com.netraedge.MainActivity { *; }
-keep class com.netraedge.SecurityHardening { *; }
-keep class com.netraedge.EncryptedAssets { *; }

# Remove Log calls in release (smaller binary + harder to reverse)
-assumenosideeffects class android.util.Log {
    public static *** d(...);
    public static *** v(...);
    public static *** i(...);
    public static *** w(...);
}


