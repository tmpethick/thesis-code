echo "=== IPOPT === Unpacking Ipopt ..." 
tar -xf Ipopt-3.11.7.tar.gz 
tar -xf coinhsl.tar.gz 

 	echo "=== buidling Ipopt ..." 
 	cd Ipopt-3.11.7/ThirdParty/ASL  && ./get.ASL 
 	cd ../../../ 
 	cd Ipopt-3.11.7/ThirdParty/Blas && ./get.Blas
 	cd ../../../ 
  	cd Ipopt-3.11.7/ThirdParty/Lapack && ./get.Lapack 
  	cd ../../../ 
  	cd Ipopt-3.11.7/ThirdParty/Metis && ./get.Metis 
 	cd ../../../   	  	
 	cd Ipopt-3.11.7/ThirdParty/HSL  && cp -r ../../../coinhsl .
  	cd ../../../ 
  	cd Ipopt-3.11.7
  	mkdir -p build_rice && 
  	cd build_rice && ../configure coin_skip_warn_cxxflags=yes
  	make -j12
  	echo "=== make IPOPT === Finished building Ipopt!" 
	make test
  	echo "=== Finished testing Ipopt!"  
  	make install 
  	echo "=== Finished installing Ipopt!" 
  	cd ../../
  	cp IpReturnCodes_SimonS.inc Ipopt-3.11.7/build_rice/include/coin
  	echo "IPOPT installed on RICE"
  	

