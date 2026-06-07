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

fullscreen = 1

[buildozer]

log_level = 2
warn_on_root = 1
