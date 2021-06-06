#!/bin/bash
for dataset in PROTEINS MUTAG COX2 DHFR BZR FRANKENSTEIN IMDB-BINARY IMDB-MULTI NCI1 NCI109
do
	oarsub -S "./create_st.sh $dataset"
done
