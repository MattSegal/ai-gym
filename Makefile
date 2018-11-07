test:
	pytest -vv

install:
	pip3 install -r requirements.txt

frozen_lake:
	python3 -m src.frozen_lake
