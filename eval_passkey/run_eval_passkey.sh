###### schedule #####
python eval_passkey/eval_passkey.py \
       --min-tokens 128 --max-tokens 8192 \
       --output-file eval_passkey/results/passkey.csv \
       --iterations 1000 --cuda --device 0
