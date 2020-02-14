black:
	black --config black.toml .

lint:
	flake8 sylph tests
	mypy --check-untyped-defs --config-file=tox.ini sylph

test:
	pytest --ignore sylph/vendor --ignore submodules

ipython:
	ipython -i scripts/python_session_init.py
