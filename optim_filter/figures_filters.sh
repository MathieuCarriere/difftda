#!/bin/bash
#for digit1 in `seq 0 8`
#do
#	for digit2 in `seq "$((digit1+1))" 9`
#	do
#		python figures_filters.py vs$digit1$digit2 0
#	done
#done
python figures.py all 0
for dataset in PROTEINS MUTAG COX2 DHFR BZR FRANKENSTEIN IMDB-MULTI IMDB-BINARY NCI1 NCI109
do
	python figures_filters.py $dataset 1
done
