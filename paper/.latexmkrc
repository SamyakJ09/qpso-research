# latexmk configuration for minted + biber
$pdf_mode = 1;                          # use pdflatex
$out_dir  = 'build';                    # keep build artefacts out of source tree
$pdflatex = 'pdflatex -shell-escape -interaction=nonstopmode %O %S';
$biber    = 'biber --output-directory=build %O %S';

# Ensure biber runs for biblatex
$bibtex_use = 2;

# Clean up minted cache
push @generated_exts, '_minted-%R/*';
$clean_ext .= ' run.xml';
