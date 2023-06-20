CXX=g++
CPPFLAGS=-g -Wall -I/opt/rocm-5.2.3/include/
LDLIBS=-g -L/opt/rocm-5.2.3/lib -lrocm_smi64

TARGET = gpu-energy

all: $(TARGET)

$(TARGET): $(TARGET).cpp
	$(CXX) $(CPPFLAGS) -o $(TARGET) $(TARGET).cpp $(LDLIBS)
