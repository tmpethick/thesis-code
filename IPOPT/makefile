PLATFORM ?= linux
COMPILER ?= gnu
REAL ?= double
DEBUG ?=
APRUN ?=
PROFILING ?= yes

CC ?= gcc
FC ?= gfortran
MPICC ?= mpicc
MPICXX ?= mpicxx
MPIF77 ?= mpif77
GPUARCH ?= 30
OPT := -rdynamic -fPIC -Dreal=$(REAL) -Dreal4=$(REAL)4 -DMPI_real=MPI_$(shell echo $(REAL) | tr a-z A-Z)
PROFILING_OPT :=
FOPT :=
GPUOPT := -arch=sm_$(GPUARCH)
ifeq (,$(DEBUG))
OPT += -pipe -g -O3 -gdwarf-3
FOPT += -finit-local-zero
GPUOPT += -use_fast_math
else
OPT += -pipe -g -O0 -gdwarf-3
GPUOPT += -G
endif

ROOT := $(shell pwd)
TARGET := $(PLATFORM)-$(COMPILER)
INSTALL := $(ROOT)/install/$(PLATFORM)/$(COMPILER)
BUILD := build/$(PLATFORM)/$(COMPILER)
IPOPT := $(ROOT)/Ipopt-3.11.7

COMMA :=,

#
# Check if CUDA Toolkit is available. If not - fallback to CPU-only
# compilation.
#
NVCC := $(shell which nvcc)

ifeq (,$(NVCC))
$(info No nvcc found, not using GPUs)
endif

#
# Use runtime optimization - compile code in runtime, optimized agressively
# with information known from runtime context, e.g. dim value.
# Runtime optimization depends on pstreams/pstream.h.
#
RUNTIME_OPTIMIZATION ?= YES

#
# Use AVX-based vectorization on CPU. Do not use this, if your CPU does not support AVX.
#
AVX ?= yes

ifeq ($(TARGET), linux-gnu)
	CC := gcc
	FC := gfortran
	#
	# We are filtering out -Werror=format-security option, because it
	# breaks ipopt compilation. 
	#
	MPICC := $(filter-out -Werror=format-security, $(shell mpicc -show))
	MPICXX := $(filter-out -Werror=format-security, $(shell mpicxx -show))
	MPIF77 := $(filter-out -Werror=format-security, $(shell mpif77 -show))
	MPILD := mpicxx
endif

ifeq ($(TARGET), linux-intel)
	CC := icc
	FC := ifort
	MPICC := mpicc
	MPICXX := mpicxx
	MPIF77 := mpif77
	MPILD := mpicxx
endif

ifeq ($(PLATFORM), cray)
        DYNAMIC := -dynamic
	CC := cc $(DYAMIC)
	FC := ftn $(DYNAMIC)
	MPICC := cc $(DYNAMIC)
	MPICXX := cc $(DYNAMIC)
	MPIF77 := ftn $(DYNAMIC)
	MPILD := cc $(DYNAMIC)
	APRUN := aprun -cc none
ifneq (,$(PROFILING))
	OPT += -fno-omit-frame-pointer
	PROFILING_OPT += -finstrument-functions -finstrument-functions-exclude-function-list=__do_global_ctors_aux,_dl_,dlopen,_mm_,_mm256
endif
endif

ifeq ($(COMPILER), gnu)
	#FLTO := -flto=jobserver
	ifeq (double,$(REAL))
		FOPT += -fdefault-real-8
	endif
	ifeq (,$(DEBUG))
		OPT += -ffast-math -funroll-loops -mtune=native -march=native -mfpmath=sse $(FLTO) -ffunction-sections -fdata-sections
	ifeq (,$(PROFILING))
		OPT += -fomit-frame-pointer
        endif
	endif
	FOPT += -Jsolver/build
	FINC := -I$(INSTALL)/include/coin/ 
	CINC := -I. -Isolver -Isparsegrid/include -Isparsegrid/Post -Isolver/sparse_grid -Itbb/include -Icuda_launch_config -Ithrust
	ifneq (,$(NVCC))
		CINC += -DHAVE_CUDA -DDEVICE_CONST_X_MAX_LENGTH=4096 -I$(shell dirname $(NVCC))/../include
	endif
	ifneq (,$(RUNTIME_OPTIMIZATION))
		CINC += -DHAVE_RUNTIME_OPTIMIZATION
	endif
	ifneq (,$(AVX))
		CINC += -DHAVE_AVX
		OPT += -mavx
	endif

	#
	# Locate folder with mpi.h and add it to include paths
	# (needed for non-MPI compiler that uses MPI-aware headers, e.g. CUDA).
	#
	CINC += -I$(shell dirname $(shell echo "\#include <mpi.h>" | $(MPICC) -xc - -M | grep -m 1 -o -e "[^ ]*mpi\.h"))
endif

ifeq ($(PLATFORM), mac)
	#
	# Need to put empty quotes after -i in BSD sed (Mac).
	#
	BSD_SED_EMPTYQUOTES := ""

	#
	# Mac may not support -pipe. Disable it for some compile time penalty.
	#
	MAC_OPT := $(filter -pipe, $(OPT))
	OPT := MAC_OPT
endif

NOPROFILING_OPT := $(OPT)
OPT += $(PROFILING_OPT)

all: $(INSTALL)/bin/hddm-solver 

#######################################################################################################################
#######################################################################################################################

ifeq (float, $(REAL))
Ipopt:
	echo "=== hddm-solver === Unpacking Ipopt ..." && \
	tar -xf $(IPOPT).tar.gz && ln -sf $(IPOPT) Ipopt && \
	patch -p1 < Ipopt.float.patch
endif
ifeq (double, $(REAL))
Ipopt:
	echo "=== hddm-solver === Unpacking Ipopt ..." && \
	tar -xf $(IPOPT).tar.gz && ln -sf $(IPOPT) Ipopt
endif

$(INSTALL)/include/coin/IpReturnCodes_SimonS.inc: Ipopt
	echo "=== hddm-solver === Buidling Ipopt ..." && \
	tar -xf $(IPOPT).tar.gz && ln -sf $(IPOPT) Ipopt && \
	cd $(IPOPT)/ThirdParty/ASL && ./get.ASL && cd $(ROOT) && \
	cd $(IPOPT)/ThirdParty/Blas && ./get.Blas && cd $(ROOT) && \
	cd $(IPOPT)/ThirdParty/HSL && tar -xf $(ROOT)/coinhsl.tar.gz && cd $(ROOT) && \
	cd $(IPOPT)/ThirdParty/Lapack && ./get.Lapack && cd $(ROOT) && \
	cd $(IPOPT)/ThirdParty/Metis && ./get.Metis && cd $(ROOT) && \
	cd $(IPOPT)/ && mkdir -p $(BUILD) && cd $(BUILD) && $(IPOPT)/configure --disable-shared --enable-static MPICC="$(MPICC)" CFLAGS="$(NOPROFILING_OPT) -ldl -lstdc++" MPICXX="$(MPICXX)" CXXFLAGS="$(NOPROFILING_OPT) -ldl -lstdc++" MPIF77="$(MPIF77) -ldl -lstdc++" FFLAGS="$(NOPROFILING_OPT) -ldl -lstdc++" LDFLAGS="-fno-lto" --prefix=$(INSTALL) && $(MAKE) && $(MAKE) install && cd $(ROOT) && \
	cp $(INSTALL)/include/coin/IpReturnCodes.inc $(INSTALL)/include/coin/IpReturnCodes_SimonS.inc && \
	sed -i $(BSD_SED_EMPTYQUOTES) 's/^C/\!/' $(INSTALL)/include/coin/IpReturnCodes_SimonS.inc && \
	echo "=== hddm-solver === Finished building Ipopt!"

#######################################################################################################################
#######################################################################################################################

GENERATION_DEPS := \
	$(INSTALL)/include/coin/IpReturnCodes_SimonS.inc \

ifeq ($(TARGET), linux-gnu)
ifneq (,$(NVCC))
$(INSTALL)/bin/hddm-solver: $(GENERATION_DEPS) 

else
$(INSTALL)/bin/hddm-solver: $(GENERATION_DEPS) 

endif
endif

ifeq ($(TARGET), mac-gnu)
ifneq (,$(NVCC))
$(INSTALL)/bin/hddm-solver: $(GENERATION_DEPS) 

else
$(INSTALL)/bin/hddm-solver: $(GENERATION_DEPS)

endif
endif 

ifeq ($(TARGET), cray-gnu)
ifneq (,$(NVCC))
$(INSTALL)/bin/hddm-solver: $(GENERATION_DEPS) 

else
$(INSTALL)/bin/hddm-solver: $(GENERATION_DEPS) 

endif
endif
  

#######################################################################################################################
#######################################################################################################################

clean:
	rm -rf $(IPOPT) Ipopt 

#######################################################################################################################
#######################################################################################################################