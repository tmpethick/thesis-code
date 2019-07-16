# Install IPOPT
cd IPOPT
./configure_rice.sh
cd ..

# Download and install pyipopt
rm -rf pyipopt
git clone https://github.com/xuy/pyipopt
cd pyipopt
sed -i "s|IPOPT_DIR = '/usr/local/'|IPOPT_DIR = '$IPOPT_DIR'|" setup.py
sed -i "s/, 'coinblas',//" setup.py
sed -i "s/libraries=\[/libraries\=\['ipopt', 'coinhsl', 'coinmetis', 'coinlapack', 'coinblas'\],/" setup.py
sed -i "51,56d" setup.py
python setup.py install
cd ..
