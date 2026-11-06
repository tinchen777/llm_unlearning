# KL + small coefficient on other layers (near_line/15: is 7-9 the right place for a small c?)

Independent scripts: run each in its own process / on its own GPU. Only `0_select_small_c.sh` has to finish first (it makes the layer selections
of c1 and c1.5, which `_ks` does not have yet). All of them use the fast evaluation profile, `increment` + `sequential` + answer tokens,
the r_uv.pt of `single_v2/near_tofu_<MODEL>_forget01_ks`, K=5, and skip runs that are already evaluated. Output: `saves/NEAR/single_v2/` and
`saves/eval/NEAR/single_v2/` (run names `rw1_ans[_kl<w>[_s<steps>]]_increment_seq_c<c>_l<layers>`).

    bash scripts/NEAR/kl_layers/0_select_small_c.sh 0      # first (inference only, a few minutes)
    bash scripts/NEAR/kl_layers/1_l9-11.sh 0               # then any of 1-3 / 4 in parallel
    bash scripts/NEAR/kl_layers/2_l10-12.sh 1
    bash scripts/NEAR/kl_layers/3_l11-13.sh 2
    bash scripts/NEAR/kl_layers/4_kl_floor_steps300.sh 3   # does the KL term still go down with more steps? (it ends at ~0.5 in every run so far)

Each of 1-3: c {1, 1.5, 2} x KL {off, 1} on one layer set = 6 runs (3 of them are the "off" controls). `DRY_RUN=1` prints the runs.
Read: forget_Q_A_Prob / mia_loss / extraction_strength, model_utility and retain ROUGE, ud_wb_unsup, ud_bb and the first words of forget_responses.json.
