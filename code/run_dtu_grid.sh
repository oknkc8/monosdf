CUDA_VISIBLE_DEVICES=5 \
python -m torch.distributed.launch \
--master_port 29425 \
--nproc_per_node 1 \
--nnodes=1 \
--node_rank=0 \
training/exp_runner.py \
--conf confs/dtu_grids_3views_reg_no_warpreg.conf  \
--exps_folder tmp_figure_exps_ablation_dtu \
--nepoch 5000 \
--scan_id 24