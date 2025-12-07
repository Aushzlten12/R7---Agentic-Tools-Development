# Makefile
IMAGE_NAME = agentic-r7

setup:
	pip install -r requirements.txt

test:
	pytest test/

run:
	python src/main.py

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run -v $(PWD)/logs:/app/logs $(IMAGE_NAME)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf logs/*