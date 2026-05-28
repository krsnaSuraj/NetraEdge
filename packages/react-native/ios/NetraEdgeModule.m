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

RCT_EXTERN_METHOD(close)

@end
