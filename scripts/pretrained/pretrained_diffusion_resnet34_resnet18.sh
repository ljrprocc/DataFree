python datafree_kd.py \
--method pretrained \
--pretrained_mode diffusion \
--pretrained_G_weight /tmp/openai-2022-07-02-15-15-09-089368/samples_100000x32x32x3.npz \
--dataset cifar10 \
--batch_size 256 \
--teacher vgg11 \
--student resnet18 \
--lr 0.1 \
--epochs 250 \
--kd_steps_interval 5 \
--ep_steps 400 \
--g_steps_interval 1 \
--T 20 \
--act 0.001 \
--balance 20 \
--gpu 1 \
--seed 20 \
--log_tag pretrained_diffusion_offline_vgg_3 \
--curr_option none \
--data_root /data/lijingru/cifar10/ 
# --pretrained_G_weight /tmp/openai-2022-07-02-15-15-09-089368/samples_100000x32x32x3.npz \
# --pretrained_G_weight /tmp/openai-2022-07-16-14-13-13-216634/samples_100000x32x32x3.npz \