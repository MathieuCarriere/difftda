#!/bin/bash
for digit1 in `seq 0 8`
do
	for digit2 in `seq "$((digit1+1))" 9`
	do
		for learning in 0.001
		do
			for batch in 150
			do
				./optim_filters.sh vs$digit1$digit2 $learning $batch 3000 0 SW 10 30 0 0
			done
		done	
	done
done
for dataset in all
do
	for learning in 0.001
	do
		for batch in 150
		do
			./optim_filters.sh $dataset $learning $batch 3000 0 SW 10 30 0 0
		done
	done	
done
for dataset in PROTEINS MUTAG COX2 DHFR BZR FRANKENSTEIN IMDB-MULTI IMDB-BINARY NCI1 NCI109
do
	for learning in 0.001
	do
		for batch in 150
		do
			for fold in `seq 0 9`
			do
				./optim_filters.sh $dataset $learning $batch 3000 10 SW 10 30 0 $fold
			done
		done
	done	
done

