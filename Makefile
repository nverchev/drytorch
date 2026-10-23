.PHONY: docs docs-fast docs-clean

# Build the documentation as Read the Docs does, from clean tutorial data.
docs: docs-clean
	DRYTORCH_INIT_MODE=minimal uv run sphinx-build -b html docs docs/_build/html

# Build without running the notebooks.
docs-fast:
	DRYTORCH_INIT_MODE=minimal uv run sphinx-build -b html \
		-D nb_execution_mode=off docs docs/_build/html

# Remove the experiment data the tutorials write and the notebook cache.
docs-clean:
	rm -rf docs/tutorials/experiments docs/jupyter_cache docs/_build
