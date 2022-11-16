CUDA_VISIBLE_DEVICES=0 \
python evaluate_single_scene.py \
--input_mesh ../exps_ablation/dtu_grids_3views_warp_occlusion_reg_83/2022_11_16_03_35_51_monosdf/plots/surface/surface_005000.ply \
--scan_id 83 \
--output_dir dtu_scn83/2022_11_16_03_35_51_monosdf