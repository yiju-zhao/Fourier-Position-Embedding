##### Books #####
#### length-512 ####
torchrun --nproc_per_node=4 scripts/train.py configs/books/length-512/ce-fourier/OLMo-60M-ce-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero.yaml

#### length-1024 ####
torchrun --nproc_per_node=4 scripts/train.py configs/books/length-1024/ce-fourier/OLMo-60M-ce-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero.yaml

##### C4 #####
#### length-512 ####
torchrun --nproc_per_node=4 scripts/train.py configs/c4/length-512/ce-fourier/OLMo-60M-ce-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero.yaml

#### length-1024 ####
torchrun --nproc_per_node=4 scripts/train.py configs/c4/length-1024/ce-fourier/OLMo-60M-ce-fourier-eye_xavier_norm_0_4-sep_basis_head-ignore_clamp_zero.yaml
