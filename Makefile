.PHONY: train api dashboard test lint docker

PYTHON ?= python

train:
	$(PYTHON) scripts/evaluate_openml_benchmark.py

api:
	$(PYTHON) -m uvicorn api.app:app --host 127.0.0.1 --port 8000

dashboard:
	$(PYTHON) -m streamlit run streamlit_app/app.py

lint:
	$(PYTHON) -m ruff check .

test: lint
	$(PYTHON) -m pytest -q

docker:
	docker compose up --build
