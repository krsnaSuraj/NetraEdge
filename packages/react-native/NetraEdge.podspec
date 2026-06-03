require 'json'

package = JSON.parse(File.read(File.join(__dir__, '..', 'package.json')))

Pod::Spec.new do |s|
  s.name         = 'NetraEdge'
  s.version      = package['version']
  s.summary      = package['description']
  s.homepage     = 'https://github.com/nhai/NetraEdge'
  s.license      = 'MIT'
  s.author       = { 'NetraEdge Team' => 'netraedge@example.invalid' }
  s.source       = { :git => 'https://github.com/nhai/NetraEdge.git', :tag => s.version }

  s.platform     = :ios, '12.0'
  s.swift_version = '5.0'

  s.source_files = 'ios/**/*.{swift,m,h}'
  s.requires_arc = true

  s.dependency 'React-Core'
  s.dependency 'TensorFlowLiteSwift', '~> 2.14'
end
