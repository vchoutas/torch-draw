# Run model with attention
th main.lua --optimizer adam --log_file attention.txt --maxEpochs 30 --read_size 5 --write_size 5 \
  --latent_size 10 --batch_size 64 --lr 1e-3 --save_image --num_glimpses 10 --use_attention true

# Run model without attention
th main.lua --optimizer adam --log_file no_attention.txt --maxEpochs 30 --read_size 5 --write_size 5 \
  --latent_size 10 --batch_size 64 --lr 1e-3 --save_image --num_glimpses 10 --use_attention false
