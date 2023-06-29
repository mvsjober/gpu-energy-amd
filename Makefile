CXX=/opt/cray/pe/gcc/12.2.0/bin/g++
CPPFLAGS=-g -Wall -I/opt/rocm-5.2.3/include/ \
	-I/opt/cray/pe/mpich/8.1.23/ofi/gnu/9.1/include/
LDFLAGS=-g \
	-Wl,-rpath,/opt/cray/pe/mpich/8.1.23/ofi/gnu/9.1/lib \
	-Wl,-rpath,/opt/cray/libfabric/1.15.2.0/lib64/ \
	-L/opt/rocm-5.2.3/lib -lrocm_smi64 \
	-L/opt/cray/pe/mpich/8.1.23/ofi/gnu/9.1/lib -lmpi \
	-L/opt/cray/libfabric/1.15.2.0/lib64/ -lfabric

TARGET = gpu-energy

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CXX) $(CPPFLAGS) -o $(TARGET) $(TARGET).cpp $(LDFLAGS)
