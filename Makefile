TEST_PATH=./tests

test:
	PYTHONPATH=. py.test --verbose --color=yes $(TEST_PATH)

push:
	ssh simba mkdir -p mthesis
	ssh simba mkdir -p mthesis/artifacts
	rsync -avu ./data pethick@simba.epfl.ch:~/mthesis
	rsync -avu ./src pethick@simba.epfl.ch:~/mthesis
	rsync -av ./notebook_header.py pethick@simba.epfl.ch:~/mthesis
	rsync -av ./explorer_helper.py pethick@simba.epfl.ch:~/mthesis
	rsync -av ./runner.py pethick@simba.epfl.ch:~/mthesis
	rsync -av ./hpc.sh pethick@simba.epfl.ch:~/mthesis

push-dtu:
	ssh s144448@login2.hpc.dtu.dk mkdir -p mthesis
	ssh s144448@login2.hpc.dtu.dk mkdir -p mthesis/artifacts
	rsync -avu ./data s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -avu ./src s144448@login2.hpc.dtu.dk:~/mthesis
	#rsync -av ./SparseGridCode s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./notebook_header.py s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./explorer_helper.py s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./runner.py s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./hpc-dtu.sh s144448@login2.hpc.dtu.dk:~/mthesis

.PHONY: test
