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

sync-dtu:
	ssh s144448@login2.hpc.dtu.dk mkdir -p mthesis
	ssh s144448@login2.hpc.dtu.dk mkdir -p mthesis/artifacts
	rsync -avu ./data s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -avu ./src s144448@login2.hpc.dtu.dk:~/mthesis
	#rsync -av ./SparseGridCode s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./notebook_header.py s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./explorer_helper.py s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./runner.py s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./hpc-dtu.sh.template s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./run_growth_model.sh s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./run_growth_model.py s144448@login2.hpc.dtu.dk:~/mthesis

	rsync -av ./install_growth.sh s144448@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./setup_env.sh s144448@login2.hpc.dtu.dk:~/mthesis
	#rsync -av ./IPOPT s144448@login2.hpc.dtu.dk:~/mthesis # Copied from asgp! don't destroy!
	rsync -av s144448@login2.hpc.dtu.dk:~/mthesis/output ./

run-growth: sync-dtu
	ssh s144448@login2.hpc.dtu.dk 'source /etc/profile; cd ~/mthesis; bsub < run_growth_model.sh'

.PHONY: test
