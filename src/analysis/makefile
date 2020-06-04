# Makefile
# requires GNUMake32
# Installation instructions: http://tilburgsciencehub.com/setup

analyze: ../../gen/analysis/output/analysis.html
preclean: ../../gen/analysis/temp/preclean.csv

../../gen/analysis/temp/preclean.csv: preclean.py ../../gen/data-preparation/output/dataset.csv
	python preclean.py


../../gen/analysis/output/analysis.html: analysis.ipynb ../../gen/analysis/temp/preclean.csv
	jupyter nbconvert --execute analysis.ipynb --output-dir=../../gen/analysis/output



# ../../gen/analysis/temp/preclean.csv: preclean.R ../../gen/data-preparation/output/dataset.csv
# 	R --vanilla --args "" < "preclean.R"


# ../../gen/analysis/output/analysis.html: analysis.py ../../gen/analysis/temp/dataset.csv
# 	R -e "rmarkdown::render('analysis.Rmd', output_file = '../../gen/analysis/output/analysis.html')"

wipe:
	python wipe.py