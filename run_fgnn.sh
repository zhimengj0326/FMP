#!/bin/bash
### lambda1 = 1000 5000 8000 10000        fairness
### lambda2 = 3 5 9 10 15 20        smoothness
### num_gnn_layer = 5 10 15  number of layers
### data = pokec_n pokec_z nba

for dataset in pokec_n
do
    for lambda2 in 0. 0.01 0.1 0.5 1.0 2.0 3 5 10 15 20 ## 0. 0.01 0.1 0.5 1.0 2.0 3 5 10 15 20 
    do
        for num_gnn_layer in 2 5 ###2 5 10 15
        do
            for lambda1 in 5 15 20 30  ### 0 5 10 15 20 30 100 1000 5000
            do
                python main.py --gpu 0 --seed=42 --epochs=300 --lambda1=$lambda1  --lambda2=$lambda2 \
                        --num_gnn_layer $num_gnn_layer --dataset=$dataset --num-hidden=64 &
            done
            # wait
        done
        wait
    done
    # wait
done


# for dataset in pokec_n
# do
#     for num_gnn_layer in 5 15
#     do
#         for lambda1 in 0 10 100 ##1000 5000
#         do
#             for lambda2 in 10 15 30 
#             do
#                 python main.py --gpu 0 --seed=42 --epochs=300 --lambda1=$lambda1  --lambda2=$lambda2 \
#                         --num_gnn_layer $num_gnn_layer --dataset=$dataset --num-hidden=64 &
#             done
#             wait
#         done
#     done
# done

# for dataset in pokec_z
# do
#     for num_gnn_layer in 5 15
#     do
#         for lambda1 in 0 10 100 ##1000 5000
#         do
#             for lambda2 in 10 15 30 
#             do
#                 python main.py --gpu 1 --seed=42 --epochs=300 --lambda1=$lambda1  --lambda2=$lambda2 \
#                         --num_gnn_layer $num_gnn_layer --dataset=$dataset --num-hidden=64 &
#             done
#             wait
#         done
#     done
# done

# python node_base.py --gpu 3 --seed=42 --epochs=10 --model=GCN --dataset=pokec_n --num-hidden=64
