# Windows Build Notes

Tested environment:

- RTX 5060 Ti 16 GB / SM120
- CUDA 12.8
- OptiX 9.1.0
- CMake 4.4.2
- Visual Studio 2022 BuildTools

Build externally when the ComfyUI path contains parentheses, because CUDA device-linking was unreliable inside `ComfyUI (1)`.

```powershell
Remove-Item -Recurse -Force "E:\Img_Gen\sol_rt_build" -ErrorAction SilentlyContinue

cmd /c 'call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" && cmake -S "E:\Img_Gen\ComfyUI-Latest\ComfyUI (1)\ComfyUI\custom_nodes\ComfyUI-sol-attn-RT-v3.2-INT8-OVERLAP\sol_rt_native" -B "E:\Img_Gen\sol_rt_build" -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=120 -DOPTIX_ROOT="C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0" && cmake --build "E:\Img_Gen\sol_rt_build"'
```

Copy the outputs back into the node:

```powershell
Copy-Item "E:\Img_Gen\sol_rt_build\sol_rt_router.dll" ".\sol_rt_native\sol_rt_router.dll" -Force
Copy-Item "E:\Img_Gen\sol_rt_build\rt_programs.ptx" ".\sol_rt_native\rt_programs.ptx" -Force
```

Useful OptiX include order:

```cpp
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
```

Verify the async export from a VS developer shell:

```powershell
dumpbin /exports sol_rt_router.dll | findstr sol_rt_route_async
```
