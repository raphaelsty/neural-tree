livedoc:
	mkdocs build --clean
	mkdocs serve --dirtyreload

deploydoc:
	mkdocs gh-deploy --force
