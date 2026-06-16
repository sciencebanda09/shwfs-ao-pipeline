sim:
	python pipeline.py --config config.yaml --mode sim

train:
	python pipeline.py --config config.yaml --mode train

eval:
	python pipeline.py --config config.yaml --mode eval

demo:
	python pipeline.py --config config.yaml --mode demo

all:
	python pipeline.py --config config.yaml --mode all

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/ models/ results/
