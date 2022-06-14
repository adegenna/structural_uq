# CMD = pandoc -V fontsize=11pt
CMD = pdflatex
BIB = bibtex
DOC = notes
BUILD = ../build

.PHONY: all view

all : Preproposal

read :
	evince $(BUILD)/$(DOC).pdf &

Preproposal : $(DOC).tex
	$(CMD) $(DOC).tex
	$(BIB) $(DOC)
	$(CMD) $(DOC).tex
	$(CMD) $(DOC).tex
	mkdir -p $(BUILD)
	mv $(DOC).aux $(DOC).bbl $(DOC).blg $(DOC).log $(DOC).pdf $(BUILD)

clean :
	rm -f *~ $(BUILD)/*
