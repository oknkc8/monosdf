CUDA_VISIBLE_DEVICES=0 \
python evaluate_single_scene.py \
--input_mesh ../exps_ablation/dtu_grids_3views_warp_occlusion_reg_106/2022_11_17_06_43_02_monosdf/plots/surface/surface_005000.ply \
--scan_id 106 \
--output_dir dtu_scn106/2022_11_17_06_43_02_monosdf