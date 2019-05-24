
COUNTER=500
I=0
         while [  $COUNTER -lt 20001 ]; do
             echo The counter is $COUNTER
             #while [ $I -lt 5 ]; do
             	./countsort_bench $COUNTER 42 >> stats_countsort_bench.csv

             let COUNTER=COUNTER+500
         done
