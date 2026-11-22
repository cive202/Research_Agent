.PHONY: install run test lint docker sample

install:
	pip install -r requirements.txt

run:
	streamlit run app.py

test:
	pytest -q

lint:
	ruff check .

sample:  ## run the app in labelled sample-data mode
	USE_SAMPLE_DATA=1 streamlit run app.py

docker:
	docker compose up --build
