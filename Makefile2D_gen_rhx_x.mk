CC			=/usr/local/bin/g++-10
# CFLAGS		=-c -Wall -fopenmp -std=c++17 #-I/usr/local/include #-I/usr/local/opt/icu4c/include #-ffast-math -Ofast -ffinite-math-only
CFLAGS		=-c -Wall -O4 -fopenmp -std=c++17 -I./#-I/Users/vaishnavi/Documents/GitHub/PhD_Vaishnavi/AIFMM_LS#-I/usr/local/include #-I/usr/local/opt/icu4c/include #-ffast-math -Ofast -ffinite-math-only
LDFLAGS		=-fopenmp -std=c++17 #-I/usr/local/include #-L/usr/local/opt/icu4c/lib #-Ofast
SOURCES		=./testFMM2D_gen_rhs_x.cpp
OBJECTS		=$(SOURCES:.cpp=.o)
DTYPE_FLAG  = -DUSE_COMPLEX64
EXECUTABLE	=./testFMM2D_gen_rhs_x

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
		$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
		$(CC) $(CFLAGS) $(DTYPE_FLAG) $< -o $@

clean:
	rm a.out testFMM2D_gen_rhs_x *.o
