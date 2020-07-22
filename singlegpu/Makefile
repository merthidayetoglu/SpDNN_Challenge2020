# ----- Make Macros -----

NVCC = nvcc
NVCCFLAGS = -lineinfo -O3 -std=c++11 -gencode arch=compute_70,code=sm_70 -Xptxas="-v" -Xcompiler -fopenmp

CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++11

TARGETS = inference
OBJECTS = main.o kernels.o

LD_FLAGS = -lgomp

# ----- Make Rules -----

all:	$(TARGETS)

%.o: %.cpp vars.h cuda_runtime.hpp
	${CXX} ${CXXFLAGS} ${OPTFLAGS} $< -c -o $@

%.o : %.cu vars.h cuda_runtime.hpp
	${NVCC} ${NVCCFLAGS} $< -c -o $@

inference: $(OBJECTS)
	$(NVCC) -o $@ $(OBJECTS) $(LD_FLAGS)

clean:
	rm -f $(TARGETS) *.o *.o.* *.txt *.bin core *.html *.xml
