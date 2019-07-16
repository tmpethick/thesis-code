# Set up the environment
export THEANO_FLAGS="gcc.cxxflags=-march=x86-64"
export PREFIX=$HOME/ext
export PATH=$PREFIX/bin:$PATH
export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONPATH=.:$PREFIX/lib/python2.7/site-packages:$PYTHONPATH
export IPOPT_DIR="`pwd`/IPOPT/Ipopt-3.11.7/build_rice"
export LD_LIBRARY_PATH=$IPOPT_DIR/lib:$LD_LIBRARY_PATH
