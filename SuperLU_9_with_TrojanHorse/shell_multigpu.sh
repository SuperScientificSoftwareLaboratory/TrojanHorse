#mpirun -n 1 ./build_gpu/EXAMPLE/pddrive -r 1 -c 1 "$MTX_A100/Ga19As19H42/Ga19As19H42.mtx" 
#mpirun -n 1 ./build_gpu/EXAMPLE/pddrive -r 1 -c 1 "$MTX_A100/Si41Ge41H72/Si41Ge41H72.mtx" 
#mpirun -n 1 ./build_gpu/EXAMPLE/pddrive -r 1 -c 1 "$MTX_A100/nlpkkt80/nlpkkt80.mtx" 
#mpirun -n 1 ./build_gpu/EXAMPLE/pddrive -r 1 -c 1 "$MTX_A100/Serena/Serena.mtx"
#
#mpirun -n 2 ./build_gpu/EXAMPLE/pddrive -r 2 -c 1 "$MTX_A100/Ga19As19H42/Ga19As19H42.mtx" 
#mpirun -n 2 ./build_gpu/EXAMPLE/pddrive -r 2 -c 1 "$MTX_A100/Si41Ge41H72/Si41Ge41H72.mtx" 
#mpirun -n 2 ./build_gpu/EXAMPLE/pddrive -r 2 -c 1 "$MTX_A100/nlpkkt80/nlpkkt80.mtx" 
#mpirun -n 2 ./build_gpu/EXAMPLE/pddrive -r 2 -c 1 "$MTX_A100/Serena/Serena.mtx"

mpirun -n 4 ./build_gpu/EXAMPLE/pddrive -r 2 -c 2 "$MTX_A100/Ga19As19H42/Ga19As19H42.mtx"
mpirun -n 4 ./build_gpu/EXAMPLE/pddrive -r 2 -c 2 "$MTX_A100/Si41Ge41H72/Si41Ge41H72.mtx"
mpirun -n 4 ./build_gpu/EXAMPLE/pddrive -r 2 -c 2 "$MTX_A100/nlpkkt80/nlpkkt80.mtx"
mpirun -n 4 ./build_gpu/EXAMPLE/pddrive -r 2 -c 2 "$MTX_A100/Serena/Serena.mtx"

#mpirun -n 8 ./build_gpu/EXAMPLE/pddrive -r 4 -c 2 "$MTX_A100/Ga19As19H42/Ga19As19H42.mtx"
#mpirun -n 8 ./build_gpu/EXAMPLE/pddrive -r 4 -c 2 "$MTX_A100/Si41Ge41H72/Si41Ge41H72.mtx"
#mpirun -n 8 ./build_gpu/EXAMPLE/pddrive -r 4 -c 2 "$MTX_A100/nlpkkt80/nlpkkt80.mtx"
#mpirun -n 8 ./build_gpu/EXAMPLE/pddrive -r 4 -c 2 "$MTX_A100/Serena/Serena.mtx"
