set -ex
python fed_flower/client.py \
    --dataroot ./datasets/facades \
    --name fed_pix2pix \
    --model pix2pix \
    --netG unet_256 \
    --direction BtoA \
    --n_epochs 10 \
    --n_epochs_decay 5 \
    --num_threads 0
        # --lambda_L1 100 \
    # --dataset_mode aligned \
    # --norm batch \
    # --pool_size 0 \
