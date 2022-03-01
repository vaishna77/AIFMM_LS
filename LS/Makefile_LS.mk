CC = /usr/local/bin/g++-9
CFLAGS		=-c -Wall -O4 -fopenmp -std=c++17 -I/usr/local/opt/icu4c/include -I/usr/local/include#-ffast-math -Ofast -ffinite-math-only
LDFLAGS		=-fopenmp -std=c++17 -L/usr/local/opt/icu4c/lib -I/usr/local/include#-Ofast
SOURCES		=./kernel.cpp  #./G_FMM2DTree_modified4.hpp #./Gauss_Legendre_Nodes_and_Weights.hpp ./panel2D.hpp ./domain2D.hpp ./gaussQuadrature.hpp ./singularNodes.hpp
OBJECTS		=$(SOURCES:.cpp=.o)
DTYPE_FLAG  = -DUSE_COMPLEX64
EXECUTABLE	=./testKernel
export CPATH = /usr/local/include
all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
		$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
		$(CC) $(CFLAGS) $(DTYPE_FLAG) $< -o $@

clean:
	rm a.out ./testKernel *.o
