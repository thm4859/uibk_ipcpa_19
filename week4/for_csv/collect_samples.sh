#initialisation
echo N, time, valid, compute_units, MFLOPs, >> stats_i5-3320M_mat_mul_seq.csv
echo N, time, valid, compute_units, MFLOPs, >> stats_i5-3320M-GPU_mat_mul_ocl.csv
echo N, time, valid, cores, MFLOPs, >> stats_i5-3320M_mat_mul_omp.csv
COUNTER=100
         while [  $COUNTER -lt 2501 ]; do
             echo The counter is $COUNTER
             ./mat_mul_ocl $COUNTER >> stats_i5-3320M-GPU_mat_mul_ocl.csv 2> /dev/null 
             ./mat_mul_omp $COUNTER >> stats_i5-3320M_mat_mul_omp.csv
             ./mat_mul_seq $COUNTER >> stats_i5-3320M_mat_mul_seq.csv
             let COUNTER=COUNTER+100
         done
