Quick guide:

For PanguLU, configure all library paths in PanguLU_5_with_TrojanHorse/make.inc and PanguLU_5_with_TrojanHorse/examples/Makefile, and run "make" in directory PanguLU_5_with_TrojanHorse to build. As built successfully, you will get executable file PanguLU_5_with_TrojanHorse/examples/pangulu_example.elf. Execute shell.sh in directory PanguLU_5_with_TrojanHorse/examples to factorise the matrix cage12.mtx.

For SuperLU, configure all library paths in SuperLU_9_with_TrojanHorse/my_build_gpu.sh, and run my_build_gpu.sh to build. As built successfully, you will get executable file SuperLU_9_with_TrojanHorse/build_gpu/EXAMPLE/pddrive. Copy the script SuperLU_9_with_TrojanHorse/shell.sh to directory SuperLU_9_with_TrojanHorse/build_gpu/EXAMPLE/, and execute shell.sh in directory SuperLU_9_with_TrojanHorse/build_gpu/EXAMPLE to factorise the matrix cage12.mtx.
