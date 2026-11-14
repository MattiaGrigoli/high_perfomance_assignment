# assignment: datamining/correlation

## commands to compile and execute
```bash
module load clang/11.0.0 cuda/10.0 #one time command
cd high_perfomance_assignment/OpenMP/datamining/correlation
make clean && make -j4 all CC=clang EXT_CFLAGS="-DLARGE_DATASET -fopenmp-targets=nvptx64-nvidia-cuda -openmp=libomp"
./correlation_acc
```

## group members
* Diego Bonatti
* Mattia Grigoli
* Etienne Ratenaina
