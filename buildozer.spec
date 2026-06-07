[app]

title = UFO Scanner
package.name = ufoscanner
package.domain = org.scripttest

source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx

version = 1.0

requirements = python3,kivy,numpy,opencv-python-headless

orientation = landscape

android.permissions = CAMERA,READ_EXTERNAL_STORAGE,WRITE_EXTERNAL_STORAGE
android.accept_sdk_license = True
android.skip_update = False
gradle_dependencies = com.android.tools.build:gradle:7.4.2
android.gradle_version = 7.4.2
gradle_options = org.gradle.jvmargs=-Xmx4096m

fullscreen = 1

[buildozer]

log_level = 2
warn_on_root = 1
