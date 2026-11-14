#!/bin/bash
# Batch 1, lane B (card 1, together with b1_seeds.sh): inference-only analyses of NEAR v2 and the 3B LUNAR fix (TODO/exp/22 #3, #4, 9c; TODO/exp/23 §4).
#   1. A1 v2 (forget05): + NEAR-Dist-v2 at 3B / 1B / 8B; + the shared-neighbour ablation at 1B / 3B (label NEAR-shared); the 3B LUNAR row re-done on the REAL forget05 point
#      (c1.5 l12; the old LUNAR.json of the misfiled forget10 model is moved to rank_attack2_3B_f05/_misfiled/). Only the new labels are queried; the summary is re-made with all models.
#   2. Naturalness attacks (1B / 3B forget05): all methods again, + NEAR-Dist-v2, 3B LUNAR fixed (CPU-light: PPL under the retain model).
#   3. Sampling consistency on forget01 (Fig. 6 of the outline): retain, full, NEAR-Act, NEAR (v2) at 1B / 3B / 8B -> saves/analysis/sample_v2_<scale>_f01/summary.md
#   4. A2 P0 (recognition baseline) with the NEAR-Dist-v2 rows and the fixed 3B LUNAR (a2_p0_lane.sh in the foreground; the 3B LUNAR activations are re-collected).
#   bash scripts/exp/b1_blackbox.sh <GPU>     (FG=1 / DRY_RUN=1 / FORCE=1)
cd "$(dirname "$0")/../.." || exit 1
G=${1:-0}; LANE=b1_blackbox; source scripts/exp/_lane.sh "$@"
A1='--forget forget05 --holdout holdout05 --extra-holdout holdout10 --templates retain --retain retain95 --k 2 --boot 200'
V4='rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1'
step a1_3b <<CMD || true
D=saves/analysis/rank_attack2_3B_f05; if [ -f \${D}/LUNAR.json ] && grep -q '_ks/c1.5_all/l14' \${D}/LUNAR.json; then mkdir -p \${D}/_misfiled; mv \${D}/LUNAR.json \${D}/_misfiled/LUNAR_forget10_ks.json; fi
python scripts/analysis/rank_attack_v2.py --gpu ${G} --out \${D} --base open-unlearning/tofu_Llama-3.2-3B-Instruct_full ${A1} --model NEAR-Dist-v2=near:saves/NEAR/forget05_Llama3B_v4/${V4}_increment_seq_c8_l19-20-21/k5_c8_all/l19-20-21 --model NEAR-shared=near:saves/NEAR/forget05_Llama3B_v4/${V4}_tshared_neighbor_increment_seq_c8_l19-20-21/k5_c8_all/l19-20-21 --model LUNAR=near:saves/LUNAR/single/lunar_tofu_Llama-3.2-3B-Instruct_forget05_LUNAR_harmful/c1.5_all/l12
CMD
step a1_1b <<CMD || true
python scripts/analysis/rank_attack_v2.py --gpu ${G} --out saves/analysis/rank_attack2_1B_f05 --base open-unlearning/tofu_Llama-3.2-1B-Instruct_full ${A1} --model NEAR-Dist-v2=near:saves/NEAR/forget05_Llama1B_v4/${V4}_increment_seq_c7_l11-12-13/k5_c7_all/l11-12-13 --model NEAR-shared=near:saves/NEAR/forget05_Llama1B_v4/${V4}_tshared_neighbor_increment_seq_c7_l11-12-13/k5_c7_all/l11-12-13
CMD
step a1_8b <<CMD || true
python scripts/analysis/rank_attack_v2.py --gpu ${G} --out saves/analysis/rank_attack2_8B_f05 --base open-unlearning/tofu_Llama-3.1-8B-Instruct_full ${A1} --model NEAR-Dist-v2=near:saves/NEAR/forget05_Llama8B_v4/${V4}_increment_seq_c7_l16-17-18/k5_c7_all/l16-17-18
CMD
step nat_1b <<CMD || true
python scripts/analysis/naturalness.py --gpu ${G} --ref-model open-unlearning/tofu_Llama-3.2-1B-Instruct_retain95 --out saves/analysis/naturalness_1B_f05.md retain=saves/eval/retain/Llama-3.2-1B-Instruct/forget05 GA=saves/tofu_unlearn/GradAscent/tofu_Llama-3.2-1B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14/evals NPO=saves/tofu_unlearn/NPO/tofu_Llama-3.2-1B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-35/evals RMU=saves/tofu_unlearn/RMU/tofu_Llama-3.2-1B-Instruct_forget05_lr5e-5_ep5_ckpt/checkpoint-28/evals LUNAR=saves/eval/LUNAR/single/lunar_tofu_Llama-3.2-1B-Instruct_forget05_LUNAR_harmful_c1.5_l11_all NEAR-Act=saves/eval/NEAR/forget05_Llama1B_v2/rw1_ans_increment_seq_c7_l7-8-9_k5_l7-8-9_c7_all NEAR-Dist=saves/eval/NEAR/forget05_Llama1B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c7_l11-12-13_k5_l11-12-13_c7_all NEAR-Dist-v2=saves/eval/NEAR/forget05_Llama1B_v4/${V4}_increment_seq_c7_l11-12-13_k5_l11-12-13_c7_all ROCR=saves/eval/ROCR/Llama-3.2-1B-Instruct_forget05_famous_l2-3-4 MEOW=saves/eval/MEOW/Llama-3.2-1B-Instruct_forget05_meow_k3_largest_lr3e-5_s200 MEOW-nb=saves/eval/MEOW/Llama-3.2-1B-Instruct_forget05_meow_nb_k3_largest_lr3e-5_s50
CMD
step nat_3b <<CMD || true
python scripts/analysis/naturalness.py --gpu ${G} --ref-model open-unlearning/tofu_Llama-3.2-3B-Instruct_retain95 --out saves/analysis/naturalness_3B_f05.md retain=saves/eval/retain/Llama-3.2-3B-Instruct/forget05 GA=saves/tofu_unlearn/GradAscent/tofu_Llama-3.2-3B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-21/evals NPO=saves/tofu_unlearn/NPO/tofu_Llama-3.2-3B-Instruct_forget05_lr2e-5_ep5_ckpt/checkpoint-14/evals RMU=saves/tofu_unlearn/RMU/tofu_Llama-3.2-3B-Instruct_forget05_lr1e-4_ep5_ckpt/checkpoint-14/evals LUNAR=saves/eval/LUNAR/single/lunar_tofu_Llama-3.2-3B-Instruct_forget05_LUNAR_harmful_c1.5_l12_all NEAR-Act=saves/eval/NEAR/forget05_Llama3B_v2/rw1_ans_increment_seq_c14_l12-13-14_k5_l12-13-14_c14_all NEAR-Dist=saves/eval/NEAR/forget05_Llama3B_v3/rw1_ans_kl1_s300_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_increment_seq_c8_l19-20-21_k5_l19-20-21_c8_all NEAR-Dist-v2=saves/eval/NEAR/forget05_Llama3B_v4/${V4}_increment_seq_c8_l19-20-21_k5_l19-20-21_c8_all ROCR=saves/eval/ROCR/Llama-3.2-3B-Instruct_forget05_famous_l3-4-5 MEOW=saves/eval/MEOW/Llama-3.2-3B-Instruct_forget05_meow_k3_largest_lr3e-5_s200 MEOW-nb=saves/eval/MEOW/Llama-3.2-3B-Instruct_forget05_meow_nb_k3_largest_lr3e-5_s200
CMD
V4A='rw1_ans_kl1_fp1_ftunseen_fmneighbor_scaled_initoriginal_kfw0_npq3_mn_rkl1'
step sample_f01_1b <<CMD || true
python scripts/analysis/sample_consistency.py --gpu ${G} --out saves/analysis/sample_v2_1B_f01 --base open-unlearning/tofu_Llama-3.2-1B-Instruct_full --forget forget01 --holdout holdout01 --model retain=hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_retain99 --model full=hf:open-unlearning/tofu_Llama-3.2-1B-Instruct_full --model NEAR-Act=near:saves/NEAR/single_v2/rw1_ans_increment_seq_c7_l7-8-9/k5_c7_all/l7-8-9 --model NEAR=near:saves/NEAR/single_v4/${V4A}_increment_seq_c1_l11-12-13/k5_c1_all/l11-12-13
CMD
step sample_f01_3b <<CMD || true
python scripts/analysis/sample_consistency.py --gpu ${G} --out saves/analysis/sample_v2_3B_f01 --base open-unlearning/tofu_Llama-3.2-3B-Instruct_full --forget forget01 --holdout holdout01 --model retain=hf:open-unlearning/tofu_Llama-3.2-3B-Instruct_retain99 --model full=hf:open-unlearning/tofu_Llama-3.2-3B-Instruct_full --model NEAR-Act=near:saves/NEAR/Llama3B_v2/rw1_ans_increment_seq_c12_l11-12-13/k5_c12_all/l11-12-13 --model NEAR=near:saves/NEAR/Llama3B_v4/${V4A}_increment_seq_c6_l19-20-21/k5_c6_all/l19-20-21
CMD
step sample_f01_8b <<CMD || true
python scripts/analysis/sample_consistency.py --gpu ${G} --out saves/analysis/sample_v2_8B_f01 --base open-unlearning/tofu_Llama-3.1-8B-Instruct_full --forget forget01 --holdout holdout01 --model retain=hf:open-unlearning/tofu_Llama-3.1-8B-Instruct_retain99 --model full=hf:open-unlearning/tofu_Llama-3.1-8B-Instruct_full --model NEAR-Act=near:saves/NEAR/Llama8B_v2/rw1_ans_increment_seq_c7_l17-18-19/k5_c7_all/l17-18-19 --model NEAR=near:saves/NEAR/Llama8B_v4/${V4A}_increment_seq_c7_l16-17-18/k5_c7_all/l16-17-18
CMD
step a2p0 <<CMD || true
rm -f logs/a2p0/done/p0_acts_3B_LUNAR; FG=1 bash scripts/exp/a2_p0_lane.sh ${G}
CMD
lane_end
