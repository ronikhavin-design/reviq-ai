# RevIQ AI: Developer workflow shortcuts
#
# Usage: make <target>
# Requires: make (Windows: install via Git for Windows or choco install make)
#
# All Python commands use the current environment's python/pip.
# Run `pip install -r requirements.txt` once before using other targets.

.PHONY: install test pipeline app api reports index docker-up

## Install all dependencies from requirements.txt
install:
	pip install -r requirements.txt

## Run the full pytest test suite
test:
	python -m pytest tests/ -v

## Run the full data-to-index pipeline (Steps 1-6)
## Generates data, trains models, scores customers, builds RAG knowledge base
pipeline:
	python run_pipeline.py

## Launch the Streamlit dashboard (requires pipeline to have run first)
app:
	streamlit run app/streamlit_app.py

## Start the FastAPI prediction service with auto-reload
api:
	uvicorn api.main:app --reload

## Regenerate only the Markdown reports from existing risk scores
reports:
	python -m src.reports.generate_reports

## Rebuild the RAG vector index from existing Markdown reports
index:
	python -m src.rag.retriever

## Build and start all services with Docker Compose
docker-up:
	docker-compose up --build
