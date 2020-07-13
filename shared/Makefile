# ----- Make Macros -----

CXX = mpicxx
CXXFLAGS = -std=c++11 -qreport -qsmp=omp -qlistfmt=html
OPTFLAGS = -O3 -qarch=pwr9 -qtune=pwr9 -qstrict -qsimd=auto

NVCC = nvcc
NVCCFLAGS = -lineinfo -O3 -std=c++11 -gencode arch=compute_70,code=sm_70 -ccbin=mpicxx -Xcompiler -qsmp=omp -Xptxas="-v"

LD_FLAGS = -ccbin=mpicxx -Xcompiler -qsmp=omp

TARGETS = inference
OBJECTS = main.o kernels.o

# ----- Make Rules -----

all:	$(TARGETS)

%.o: %.cpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} $^ -c -o $@

%.o : %.cu
	${NVCC} ${NVCCFLAGS} $^ -c -o $@

inference: $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core *.html *.xml
