nnodes=2
base_dir=RESULTS/perf_eval/${nnodes}_nodes/twin_test
mkdir -p $base_dir


## VARIABLES DEFINITION
# Model vars
IMAGES=(6982)
LEARNING_RATES=(0.001 0.0001)
BATCH_SIZE=(32 64 128 256)
# BS 16 no way
MOMENTUM=(0.9 0.99)

# Default Ddropout
: '
DROPOUT0=(0.5)
DROPOUT1=(0.5)
DROPOUT2=(0.5)
DROPOUT3=(0.5)
DROPOUT4=(0.5)
DROPOUT5=(0.5)
DROPOUT6=(0.5)
DROPOUT7=(0.5)
'


DROPOUT0=(0.3 0.4 0.5 0.6 0.7)
DROPOUT1=(0.3 0.4 0.5 0.6 0.7)
DROPOUT2=(0.3 0.4 0.5 0.6 0.7)
DROPOUT3=(0.3 0.4 0.5 0.6 0.7)
DROPOUT4=(0.3 0.4 0.5 0.6 0.7)
DROPOUT5=(0.3 0.4 0.5 0.6 0.7)
DROPOUT6=(0.3 0.4 0.5 0.6 0.7)
DROPOUT7=(0.3 0.4 0.5 0.6 0.7)

# System vars
PPN=1
INTERT=2
CPS=$((24 / PPN))
INTRAT=$((CPS-INTERT))
BLOCKTIME=(1 10)


for k in ${!BLOCKTIME[@]}; do
    BKT=${BLOCKTIME[$k]}
    
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
                        
                            for d2 in ${!DROPOUT2[@]}; do
                                D2=${DROPOUT2[$d2]}
                                
                                for d3 in ${!DROPOUT3[@]}; do
                                    D3=${DROPOUT3[$d3]}
                                    
                                    for d4 in ${!DROPOUT4[@]}; do
                                        D4=${DROPOUT4[$d4]}
    
                                        for d5 in ${!DROPOUT5[@]}; do
                                            D5=${DROPOUT5[$d5]}
                                            
                                            for d6 in ${!DROPOUT6[@]}; do
                                                D6=${DROPOUT6[$d6]}
                                                
                                                for d7 in ${!DROPOUT7[@]}; do
                                                    D7=${DROPOUT7[$d7]}
                                                    
                                                    test_name=ppn_${PPN}-cps_${CPS}-intra_${INTRAT}-inter_${INTERT}-bt_${BKT}-img_${IMG}-lr_${LR}-mom_${M}-bs_${BS}-dpout_${D0}_${D1}_${D2}_${D3}_${D4}_${D5}_${D6}_${D7}
                                                    save_path=$base_dir/$test_name

						    DIR=$save_path
                                                    if [ ! -d "$DIR" ]; then
                                                    # Take action if $DIR exists. #
	                                            
							mkdir -p $save_path

        	                                        /usr/lib64/openmpi-4.0.3/bin/mpirun --allow-run-as-root --tag-output --report-bindings --oversubscribe \
                                                            --mca orte_base_help_aggregate 0 --mca btl tcp,self --mca btl_tcp_if_include 10.20.0.0/24 \
                                                            -H 10.20.0.38,10.20.0.58 \
                                                            --map-by ppr:$PPN:socket:pe=$CPS /home/ASPIRE/shared/python_env/aspenv/bin/python \
                                                            /home/ASPIRE/shared/aspire/medical-decathlon/single-node/Train-Solution.py \
                                                            --train_images $IMG --batch_size $BS --learning_rate $LR --momentum $M \
                                                            --dropout0 $D0 --dropout1 $D1 --dropout2 $D2 --dropout3 $D3 --dropout4 $D4 --dropout5 $D5 --dropout6 $D6 --dropout7 $D7 \
                                                            --num_threads $INTRAT --blocktime $BKT --num_inter_threads $INTERT \
                                                            --output_path $save_path 2>&1 | tee $save_path/$test_name.log
                                                            
                	                                 # Run log_parser analysis
                        	                         python log_parser.py --logfile_name $test_name.log --input_path $save_path --output_path $save_path
                                	                 #  Clear PageCache only.
                                        	         sync; echo 1 > /proc/sys/vm/drop_caches
                                                	 # Clear dentries and inodes.
                                                    	 sync; echo 2 > /proc/sys/vm/drop_caches
                                                    	 #Clear PageCache, dentries and inodes
                                                    	 sync; echo 3 > /proc/sys/vm/drop_caches 
						    else
							echo "TEST_NAME found, skipping"
						        echo $test_name >> 2_NODES_TWIN_REPORT.txt
						    fi
                                                done                                      
                                            done                                                                         
                                        done   
                                    done                                                                        
                                done                                                                        
                            done                                            
                        done                    
                    done
                done              
            done
        done        
    done
done
