#!/bin/bash
#for digit1 in `seq 0 8`
#do
#	for digit2 in `seq "$((digit1+1))" 9`
#	do
#		for learning in 0.001
#		do
#			for batch in 150
#			do
#				oarsub -S "./optim_filters.sh vs$digit1$digit2 $learning $batch 3000 0 SW 10 30 0 0"
#			done
#		done	
#	done
#done
#for dataset in all
#do
#	for learning in 0.001
#	do
#		for batch in 150
#		do
#			oarsub -S "./optim_filters.sh $dataset $learning $batch 3000 0 SW 10 30 0 0"
#		done
#	done	
#done
for dataset in MUTAG PROTEINS DHFR IMDB-BINARY COX2 BZR FRANKENSTEIN IMDB-MULTI NCI1 NCI109
do
	for learning in 0.0001
	do
		for batch in 150
		do
			for fold in `seq 0 9`
			do
				oarsub -S "./optim_filters.sh $dataset $learning $batch 200 10 SW 10 30 0 $fold"
			done
		done
	done	
done

