all:
	sudo pip install -e .
clean:
	sudo pip uninstall mysplib
	sudo rm -rdf build
	sudo rm -f pysplib/*.so
	sudo rm -drf *.egg-info
	sudo rm -drf pysplib/__pycache__
test:
	sudo python3 tests/test.py

