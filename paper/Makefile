# ═════════════════════════════════════════════════════════════
# Makefile — PSO vs QPSO paper
# Requires: pdflatex, biber, latexmk, pygments (for minted)
# ═════════════════════════════════════════════════════════════

MAIN     = main
BUILDDIR = build

export TEXMF_OUTPUT_DIRECTORY = $(CURDIR)/$(BUILDDIR)

.PHONY: all clean watch

all: $(BUILDDIR)/$(MAIN).pdf

$(BUILDDIR)/$(MAIN).pdf: $(MAIN).tex preamble.tex references.bib sections/*.tex
	@mkdir -p $(BUILDDIR)
	latexmk -pdf -shell-escape -output-directory=$(BUILDDIR) $(MAIN).tex

watch:
	latexmk -pdf -shell-escape -output-directory=$(BUILDDIR) -pvc $(MAIN).tex

clean:
	latexmk -output-directory=$(BUILDDIR) -C $(MAIN).tex
	rm -rf $(BUILDDIR) _minted-$(MAIN)
