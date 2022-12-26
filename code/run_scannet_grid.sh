CUDA_VISIBLE_DEVICES=0 \
python -m torch.distributed.launch \
--master_port 29490 \
--nproc_per_node 1 \
--nnodes=1 \
--node_rank=0 \
training/exp_runner.py \
--conf confs/scannet_grids_sparseviews_reg.conf  \
--exps_folder exps_ablation_scannet_new \
--nepoch 200 \
--scan_id 2