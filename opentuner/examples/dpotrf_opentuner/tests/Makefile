C         =  gcc
CFPIC     =  -fPIC
CFLAGS    =  -fopenmp $(FPIC) -O3 -std=c99 -Wall -pedantic -Wshadow -Wno-unused
LDFLAGS   =  -fopenmp $(FPIC)
# options for MKL
CFLAGS    += -DPLASMA_WITH_MKL \
             -DMKL_Complex16="double _Complex" \
             -DMKL_Complex8="float _Complex"
#INC       =  -I../include
INC       =  -I/home/pedro/plasma_autotuner/include
#INC       += -I../test
INC       += -I/home/pedro/plasma_autotuner/test
LIBS      =  -L$(MKLROOT)/lib -lmkl_intel_lp64 -lmkl_core -lmkl_sequential -lm
INC       += -I$(MKLROOT)/include
#LIBS      += -L../lib -lplasma -lcoreblas
LIBS      += -L/home/pedro/plasma_autotuner/lib -lplasma -lcoreblas

#TILE_SIZE
CFLAGS    += -DTILE_SIZE="128" 
#CFLAGS    += -S 

all:

	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  dccrb2cm.c    -o  dccrb2cm.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  dcm2ccrb.c    -o  dcm2ccrb.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  core_dgemm.c  -o  core_dgemm.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  pdgemm.c      -o  pdgemm.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  dgemm.c       -o  dgemm.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  core_dsyrk.c  -o  core_dsyrk.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  pdsyrk.c      -o  pdsyrk.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  dsyrk.c       -o  dsyrk.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  core_dtrsm.c  -o  core_dtrsm.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  pdtrsm.c      -o  pdtrsm.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  dtrsm.c       -o  dtrsm.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  core_dpotrf.c  -o  core_dpotrf.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  pdpotrf.c      -o  pdpotrf.o 
	$(CC) -c  $(INC)  $(CFLAGS)  $(LIBS)  dpotrf.c       -o  dpotrf.o 
	$(CC) dcm2ccrb.o dccrb2cm.o core_dgemm.o pdgemm.o dgemm.o core_dsyrk.o pdsyrk.o dsyrk.o core_dtrsm.o pdtrsm.o dtrsm.o core_dpotrf.o pdpotrf.o dpotrf.o $(INC)  $(CFLAGS)  $(LIBS)  test_opentuner.c  -o  test 


clean:
	rm *.o
	rm test
