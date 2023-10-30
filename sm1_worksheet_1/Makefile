.DEFAULT_GOAL := worksheet_01.tar.gz

worksheet/worksheet.pdf : 
	cd worksheet && latexmk -pdf worksheet.tex

worksheet_01.tar.gz :  worksheet/worksheet.pdf test files
	mkdir -p build
	cp -r templates build/solutions
	cp -r test files worksheet/worksheet.pdf build/
	cd build &&	tar --exclude-from=${CI_PROJECT_DIR}/.excludes -czvf worksheet_01.tar.gz test files worksheet.pdf solutions
	mv build/worksheet_01.tar.gz .

.PHONY : clean
clean :
	rm worksheet_01.tar.gz
	rm -r build
