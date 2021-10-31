VENV := venv
BIN := $(VENV)/bin
PYTHON := $(BIN)/python
SHELL := /bin/bash

.PHONY: venv
venv: # make virtual enviroment
	python3 -m venv $(VENV) && source $(BIN)/activate

.PHONY: install
install: venv requirements.txt # install requirements and change to code folder
	$(BIN)/pip install -U pip
	$(BIN)/pip install -r requirements.txt

.PHONY: clean
clean: # remove generated files
	rm -rf __pycache__
	rm -rf $(VENV)
