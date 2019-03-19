TEST_PATH=./tests

test:
	PYTHONPATH=. py.test --verbose --color=yes $(TEST_PATH)
