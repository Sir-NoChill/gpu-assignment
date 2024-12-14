# Procedure

## Gem5 in GPU Mode

In order to prepare gem5 for GPU simulation, you need to build another binary with the `scons` build system. Use the command `scons build/VEGA_X86/gem5.opt` to build the GPU simulator. This may take some time. Go grab a hot chocolate.

## Performing the experiments

- Find the `lin_reg.amdgpu` binary in the 429-resources folder as well as the corresponding `data.amdgpu.lin_reg` file.
    - The data file is a text file with points in a point cloud arranged with one x and one y value per line
- The required linux kernel to use for this simulation is in /cshome/429/resources/amdgpu_kernel
- The required disk image used in this simulation is /cshome/429/resources/amdgpu_disk
- To check if your system is set up properly, run the following command:
```
sudo gem5/build/VEGA_X86/gem5.opt gem5/configs/example/gpufs/mi200.py \
--app $429_DIR/resources/bin/lin_reg \
--disk-image /cshome/429/resources/amdgpu_disk \
--kernel /cshome/429/resources/amdgpu_kernel --no-kvm-perf \
- o $429_DIR/resources/data/data.andgpu.lin_reg
```

- Run the simulation using the script `$GEM_PATH/configs/example/gpufs/mi200.py` (as above) and record the statistic `shaderActiveTicks` and `ALUUtilization` while varying the number of compute units available in the system.
    :::tip
    Check out the cli arguments for the `mi200.py` script
    :::
- Do the same experiment while using the `lin_reg_iter` binary
