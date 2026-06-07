[app]
title = MINOS SIGINT UFO Scanner
package.name = minos_sigint
package.domain = art.kruix
source.dir = .
source.include_exts = py,png,jpg,kv,atlas,onnx
version = 1.0
requirements = python3,kivy,numpy
orientation = portrait
fullscreen = 0
android.permissions = CAMERA, INTERNET, READ_EXTERNAL_STORAGE, WRITE_EXTERNAL_STORAGE
android.api = 33
android.minapi = 24
android.archs = arm64-v8a
android.accept_sdk_license = True
p4a.branch = master
# entrypoint
# android.entrypoint = main.py
