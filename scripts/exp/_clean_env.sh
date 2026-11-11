# Sourced at the top of the lane scripts: forget the experiment variables a terminal may still have EXPORTED from an earlier command (e.g. `export FORGET_SPLIT=forget10 ...`).
# The scripts under scripts/{NEAR,LUNAR,unlearn,exp} read these with defaults (`${FORGET_SPLIT:-forget01}`), so a stale export silently changes a step that does not set them itself
# (2026-11-11: the ROCR steps of tonight_lane_c.sh were evaluated on forget10 and failed). Every step of a lane sets what it needs on its own command line.
unset MODEL FORGET_SPLIT FORGET_SPLITS NEIGHBOR_SPLIT RETAIN_SPLIT HOLDOUT_SPLIT AUTHOR_OFFSET TASK_NAME SELECT_TASK_NAME OUT_PREFIX UPSTREAM_PREFIX R_UV_PREFIX \
  EXTRA_TRAIN EXTRA_EVAL EXTRA_WDOWN EXTRA_SELECT EXTRA_ROCR EXTRA TAG LRS METHODS DTYPES COEFF COEFFS LAYER LAYERS LAYER_SETS LAYER_FRACTION INCLUDE_ANSWER MULTI_LAYER SEQUENTIAL \
  RETAIN_WEIGHTS RETAIN_LOGS_PATH EVAL_PROFILE KS_LIST MODES SOLVER K RUV_KS REFS SCALE TARGET SKIP_EDIT SEED SAMPLE_SEED SETTING VARIANTS NO_SAMPLE SKIP_TRAIN \
  KL_STEPS KL_INIT KL_WEIGHTS KL_FORGET_W KL_NPQ KL_MASK_NAME FP_WEIGHT FP_TARGET FP_MODE PROBE EPOCHS LR
