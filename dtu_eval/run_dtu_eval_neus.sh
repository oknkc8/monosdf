CUDA_VISIBLE_DEVICES=0 \
python evaluate_single_scene.py \
--input_mesh ../neus_exp/00050000_97.ply \
--scan_id 97 \
--output_dir neus/dtu_scn97 \
--sparseneus True