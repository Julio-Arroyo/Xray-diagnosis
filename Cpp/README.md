# To generate Makefile
From ./Cpp directory do
```
>>> mkdir build && cd build
>>> /home/jarroyoi/cmake-3.26.3/bin/cmake -DCUDA_INCLUDE_DIRS=/software/Modules/modulefiles/libraries/cuda/12.0 -DCMAKE_PREFIX_PATH=/groups/CS156b/libtorch_preABI/libtorch ..
```

# To compile
```
>>> make
```