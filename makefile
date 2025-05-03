
setup:
	@echo "Setting up local environment..."
	@echo "Installing poetry..."
	# @curl -sSL https://install.python-poetry.org | python3 -
	@echo "Entering poetry shell..."
	@poetry install
	@poetry shell

run:
	@echo "Running the application..."
	@python3 cluster_pdfs.py