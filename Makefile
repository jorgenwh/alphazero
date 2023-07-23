.PHONY: all install dev-install uninstall clean

all: install

install: clean
	pip install .

dev-install: clean
	pip install -e .

uninstall: clean
	pip uninstall alphazero

clean:
	$(RM) -rf __pycache__
	$(RM) -rf alphazero/__pycache__
	$(RM) -rf alphazero/games/__pycache__
	$(RM) -rf alphazero/games/tictactoe/__pycache__
	$(RM) -rf build
	$(RM) -rf alphazero.egg-info
	$(RM) -rf *.so
	$(RM) -rf dist/
