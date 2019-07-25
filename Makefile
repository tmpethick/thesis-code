TEST_PATH=./tests
USERNAME=s144448

test:
	PYTHONPATH=. py.test --verbose --color=yes $(TEST_PATH)

push:
	ssh simba mkdir -p ~/mthesis
	ssh simba mkdir -p ~/mthesis/artifacts
	rsync -avu ./data pethick@simba.epfl.ch:~/mthesis
	rsync -avu ./src pethick@simba.epfl.ch:~/mthesis
	rsync -av ./notebook_header.py pethick@simba.epfl.ch:~/mthesis
	rsync -av ./explorer_helper.py pethick@simba.epfl.ch:~/mthesis
	rsync -av ./runner.py pethick@simba.epfl.ch:~/mthesis
	rsync -av ./hpc.sh pethick@simba.epfl.ch:~/mthesis

sync-dtu:
	ssh $(USERNAME)@login2.hpc.dtu.dk mkdir -p ~/mthesis
	ssh $(USERNAME)@login2.hpc.dtu.dk mkdir -p ~/mthesis/artifacts
	rsync -avu ./data $(USERNAME)@login2.hpc.dtu.dk:~/mthesis
	rsync -avu ./src $(USERNAME)@login2.hpc.dtu.dk:~/mthesis
	#rsync -av ./SparseGridCode $(USERNAME)@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./notebook_header.py $(USERNAME)@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./explorer_helper.py $(USERNAME)@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./runner.py $(USERNAME)@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./hpc-dtu.sh.template $(USERNAME)@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./run_growth_model.sh $(USERNAME)@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./run_growth_model.py $(USERNAME)@login2.hpc.dtu.dk:~/mthesis

	rsync -av ./install_growth.sh $(USERNAME)@login2.hpc.dtu.dk:~/mthesis
	rsync -av ./setup_env.sh $(USERNAME)@login2.hpc.dtu.dk:~/mthesis
	#rsync -av ./IPOPT $(USERNAME)@login2.hpc.dtu.dk:~/mthesis # Copied from asgp! don't destroy!
	rsync -av $(USERNAME)@login2.hpc.dtu.dk:~/mthesis/output ./

run-growth: sync-dtu
	ssh $(USERNAME)@login2.hpc.dtu.dk 'source /etc/profile; cd ~/mthesis; bsub < run_growth_model.sh'

.PHONY: test
