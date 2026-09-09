NUM_GPUS=8
random_number=$((RANDOM % 100 + 1200))


accelerate launch --multi_gpu --num_processes $NUM_GPUS train.py \
    --report-to="wandb" \
    --allow-tf32 \
    --mixed-precision="fp16" \
    --seed=0 \
    --path-type="linear" \
    --prediction="v" \
    --weighting="uniform" \
    --model="SiT-XL/2" \
    --enc-type="dinov2-vit-b" \
    --proj-coeff=0.5 \
    --encoder-depth=8 \     #SiT-L/XL use 8, SiT-B use 4
    --output-dir="your_path/reg_xlarge_dinov2_base_align_8_cls" \
    --exp-name="linear-dinov2-b-enc8" \
    --batch-size=256 \
    --data-dir="data_path/imagenet_vae" \
    --cls=0.03


    #Dataset Path
    #For example: your_path/imagenet-vae
    #This folder contains two folders
    #(1) The imagenet's RGB image: your_path/imagenet-vae/imagenet_256-vae/
    #(2) The imagenet's VAE latent: your_path/imagenet-vae/vae-sd/