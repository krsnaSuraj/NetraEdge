/**
 * NetraEdgeModule — Objective-C bridge for React Native.
 *
 * React Native requires an ObjC file to register Swift native modules.
 * This file exposes the Swift NetraEdgeModule class to the RN bridge.
 */

#import <React/RCTBridgeModule.h>

@interface RCT_EXTERN_MODULE(NetraEdgeModule, NSObject)

RCT_EXTERN_METHOD(initialize:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(isInitialized:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(runRecognition:(NSArray *)pixels
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(runLiveness:(NSArray *)pixels
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(cosineSimilarity:(NSArray *)a
                  b:(NSArray *)b
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(cropFace:(NSArray *)pixels
                  width:(nonnull NSNumber *)width
                  height:(nonnull NSNumber *)height
                  x:(nonnull NSNumber *)x
                  y:(nonnull NSNumber *)y
                  w:(nonnull NSNumber *)w
                  cropH:(nonnull NSNumber *)cropH
                  resolver:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(getEmbeddingDim:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

RCT_EXTERN_METHOD(close)

RCT_EXTERN_METHOD(setUseTTA:(BOOL)useTTA)

RCT_EXTERN_METHOD(isUseTTA:(RCTPromiseResolveBlock)resolve
                  rejecter:(RCTPromiseRejectBlock)reject)

@end
