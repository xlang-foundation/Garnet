from pathlib import Path
import subprocess,concurrent.futures
w=Path(__file__).resolve().parent;s='/Users/developer/CantorAI/out/research/worldsense-primary-kmajor/source'
def build(mode):
 b='/Users/developer/CantorAI/out/build/garnet-metal-worldsense-primary-kmajor-'+mode
 with(w/('build-'+mode+'.log')).open('w')as log:
  subprocess.run(['/opt/homebrew/bin/cmake','-S',s,'-B',b,'-C','/Users/developer/CantorAI/Garnet/cmake/MetalWorldSense4BPrefix.cmake','-DGARNET_METAL=ON','-DGARNET_METAL_XMODEL=ON','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_OBJCXX_COMPILER=/usr/bin/c++','-DGARNET_METAL_LAZY_RESIDENT_DEFAULTS=OFF','-DGARNET_METAL_CACHED_MATRIX_METADATA=OFF','-DGARNET_METAL_PRIMARY_BF16_BLOCKS=ON','-DGARNET_METAL_PRIMARY_TEST_FAULTS=ON','-DXLANG3_SDK_ROOT=/Users/developer/CantorAI/xlang3/sdk','-DGARNET_XLANG3_RUNTIME_LIBRARY=/Users/developer/CantorAI/out/build/macos-arm64-release/bin/libxlang3_runtime.dylib'],stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run(['/opt/homebrew/bin/cmake','--build',b,'-j','2'],stdout=log,stderr=subprocess.STDOUT,check=True)
 print('BUILD PASS',mode,flush=True)
build('fault')
