Pod::Spec.new do |s|
  s.name             = 'NetraEdge'
  s.version          = '0.1.0'
  s.summary          = 'Offline facial recognition + liveness detection for React Native.'
  s.description      = <<-DESC
    NetraEdge provides on-device face recognition and 7-layer liveness detection
    for React Native apps. Designed to integrate with the Datalake 3.0 (NHAI)
    field-personnel attendance workflow. Bundles TFLite inference.
  DESC
  s.homepage         = 'https://github.com/netraedge/netraedge'
  s.license          = { :type => 'Apache-2.0', :file => 'LICENSE' }
  s.author           = { 'NetraEdge' => '[email protected]' }
  s.platform         = :ios, '12.0'
  s.source           = { :git => '[email protected]:netraedge/netraedge.git', :tag => s.version.to_s }
  s.source_files     = '*.{h,m,mm,swift}'
  s.requires_arc     = true
  s.swift_version    = '5.0'
  s.frameworks       = 'CoreVideo', 'AVFoundation', 'CoreImage'
  s.dependency 'React-Core'
  s.dependency 'TensorFlowLiteSwift', '~> 2.14'
  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'SWIFT_OBJC_INTERFACE_HEADER_NAME' => 'NetraEdge-Swift.h',
    'CLANG_ENABLE_MODULES' => 'YES',
  }
end
