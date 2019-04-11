
COUNTER=10
I=0
         while [  $COUNTER -lt 10000000000 ]; do
             echo The counter is $COUNTER
             #while [ $I -lt 5 ]; do
             	./count_seq $COUNTER 42 >> stats_GT_730_seq.csv
             	./faulty_count_ocl $COUNTER >> stats_GT_730_ocl.csv
             	./count_omp $COUNTER 42 >> stats_GT_730_omp.csv

             let COUNTER=COUNTER*10
         done
