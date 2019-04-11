
COUNTER=10
I=0
         while [  $COUNTER -lt 1000000000 ]; do
             echo The counter is $COUNTER
             #while [ $I -lt 5 ]; do
             	./count_seq $COUNTER 42 >> stats_i7-4710HQ_seq.csv
             	./count_ocl $COUNTER 42 >> stats_HD_Graphics_4600_ocl.csv
             	./count_omp $COUNTER 42 >> stats_i7-4710HQ_omp.csv

             let COUNTER=COUNTER*10
         done
