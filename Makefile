TEST_PATH=./tests

test:
	PYTHONPATH=. py.test --verbose --color=yes $(TEST_PATH)

push:
	ssh simba mkdir -p mthesis
	ssh simba mkdir -p mthesis/artifacts
	rsync -avu ./data pethick@simba.epfl.ch:~/mthesis
	rsync -avu ./src pethick@simba.epfl.ch:~/mthesis
	rsync -av ./runner.py pethick@simba.epfl.ch:~/mthesis
	rsync -av ./hpc.sh pethick@simba.epfl.ch:~/mthesis

.PHONY: test
