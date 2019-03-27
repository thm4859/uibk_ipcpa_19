#initialisation
echo N, time, valid, compute_units, MFLOPs, >> stats_gtx960_mat_mul_seq.csv
echo N, time, valid, compute_units, MFLOPs, >> stats_gtx960_mat_mul_ocl.csv
echo N, time, valid, cores, MFLOPs, >> stats_phenom_x6_1075T_mat_mul_omp.csv
COUNTER=10
I=0
         while [  $COUNTER -lt 5000 ]; do
             echo The counter is $COUNTER
             while [ $I -lt 5 ]; do
             	./mat_mul_ocl $COUNTER >> stats_gtx960_mat_mul_ocl.csv
             	./mat_mul_omp $COUNTER >> stats_phenom_x6_1075T_mat_mul_omp.csv
		./mat_mul_seq $COUNTER >> stats_phenom_x6_1075T_mat_mul_seq.csv
		let I=I+1
	     done
             let I=0
	     if [ $COUNTER -eq 10 ]
	     then
		let COUNTER=0
	     fi
             let COUNTER=COUNTER+100
         done
