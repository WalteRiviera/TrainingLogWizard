nnodes=2
base_dir=OUTPUTS/experiments_${nnodes}_nodes/
mkdir -p $base_dir


## VARIABLES DEFINITION
# Model vars
IMAGES=(1 10 100)
LEARNING_RATES=(0.001 0.0001)
BATCH_SIZE=(16 32 64)
MOMENTUM=(0.9 0.99)

DROPOUT0=(0.2 0.5 0.7)
DROPOUT1=(0.2 0.5 0.7)
...
DROPOUTN=(0.2 0.5 0.7)

# System vars
PPN=1
INTERT=2
CPS=$((24 / PPN))
INTRAT=$((CPS-INTERT))

for i in ${!IMAGES[@]}; do
IMG=${IMAGES[$i]}

	for l in ${!LEARNING_RATES[@]}; do
	    LR=${LEARNING_RATES[$l]}

	    for m in ${!MOMENTUM[@]}; do
		M=${MOMENTUM[$m]}

			for b in ${!BATCH_SIZE[@]}; do
				BS=${BATCH_SIZE[$b]}

				for d0 in ${!DROPOUT0[@]}; do
				D0=${DROPOUT0[$d0]}

					for d1 in ${!DROPOUT1[@]}; do
						D1=${DROPOUT1[$d1]}

						for d2 in ${!DROPOUTN[@]}; do
						DN=${DROPOUT2[$dN]}
							test_name=ppn_${PPN}-cps_${CPS}-intra_${INTRAT}-inter_${INTERT}-bt_${BKT}-img_${IMG}-lr_${LR}-mom_${M}-bs_${BS}-dpout_${D0}_${D1}_${D2}_${D3}_${D4}_${D5}_${D6}_${D7}
							save_path=$base_dir/$test_name

							DIR=$save_path
							if [ ! -d "$DIR" ]; then
							# Take action if $DIR exists. #

							mkdir -p $save_path

							mpirun --tag-output --report-bindings --oversubscribe \
								-H my_host1, my_host2 \
								--map-by ppr:$PPN:socket:pe=$CPS python my_python_training_script.py
								--train_images $IMG --batch_size $BS --learning_rate $LR --momentum $M \
								--dropout0 $D0 --dropout1 $D1 --dropoutN $DN \
								--num_threads $INTRAT --num_inter_threads $INTERT \
								--output_path $save_path 2>&1 | tee $save_path/$test_name.log

							 # Run log_parser analysis
							 python log_parser.py --logfile_name $test_name.log --input_path $save_path --output_path $save_path

							else
								echo "TEST_NAME found, skipping"
								echo $test_name >> SKIP_REPORT.txt
							fi
							done                                                                        
						done                                                                        
					done                                            
				done                    
		    done
		done              
	done        
done

