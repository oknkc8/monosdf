CUDA_VISIBLE_DEVICES=0 \
python evaluate_single_scene.py \
--input_mesh ../exps_ablation_dtu/dtu_grids_3views_warp_occlusion_reg_105/2022_12_01_18_10_39_monosdf_depth_reg_warp_pixel_from_start_warp_both_after_half_l1loss/plots/surface/surface_005000.ply \
--scan_id 105 \
--output_dir dtu_scn105/2022_12_01_18_10_39_ours2
