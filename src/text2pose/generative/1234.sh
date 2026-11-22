# 1. PPSG-Full: Full -> Full
bash generative/script_generative.sh \
  -a train-eval \
  -s 0 \
  -z ppsg_full \
  --train_text_mode full \
  --test_text_mode full

# 2. PPSG-Full: Full -> Mask
bash generative/script_generative.sh \
  -a eval \
  -s 0 \
  -z ppsg_full \
  --train_text_mode full \
  --test_text_mode mask \
  --test_mask_ratios 0.1 0.2 0.3 0.5 0.7

# 3. PPSG-Aug: Aug -> Full
bash generative/script_generative.sh \
  -a train-eval \
  -s 0 \
  -z ppsg_aug \
  --train_text_mode aug \
  --test_text_mode full \
  --train_mask_ratios 0.1 0.2 0.3 0.5 0.7

# 4. PPSG-Aug: Aug -> Mask
bash generative/script_generative.sh \
  -a eval \
  -s 0 \
  -z ppsg_aug \
  --train_text_mode aug \
  --test_text_mode mask \
  --train_mask_ratios 0.1 0.2 0.3 0.5 0.7 \
  --test_mask_ratios 0.1 0.2 0.3 0.5 0.7