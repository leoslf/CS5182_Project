all: docs

SPHINX_APIDOC = sphinx-apidoc

docs:
	$(SPHINX_APIDOC) -o $@/source/ compression -f
	$(MAKE) -C $@ html

.PHONY: all docs
