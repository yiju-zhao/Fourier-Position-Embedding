##### YARN #####
#### RoPE-512 + YARN-1024 ####
torchrun --nproc_per_node=4 scripts/train.py \
         configs/c4/length-1024/ce-extra/x+yarn/OLMo-60M-ce-yarn.yaml \
         --load_path=workspace/OLMo-60M-ce-512-c4/latest-unsharded
#### FoPE-512 + YARN-1024 ####
torchrun --nproc_per_node=4 scripts/train.py \
         configs/c4/length-1024/ce-extra/x+yarn/OLMo-60M-ce-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero-yarn_after.yaml \
         --load_path=workspace/OLMo-60M-ce-512-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero-c4/latest-unsharded

##### FoPE #####
#### RoPE-512 + FoPE-1024 ####
torchrun --nproc_per_node=4 scripts/train.py \
         configs/c4/length-1024/ce-extra/x+fope/OLMo-60M-ce-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero-extra.yaml \
         --load_path=workspace/OLMo-60M-ce-512-c4/latest-unsharded

##### FoPE-512 + FoPE-1024 #####
torchrun --nproc_per_node=4 scripts/train.py \
         configs/c4/length-1024/ce-extra/x+fope/OLMo-60M-ce-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero-extra.yaml \
         --load_path=workspace/OLMo-60M-ce-512-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero-c4/latest-unsharded