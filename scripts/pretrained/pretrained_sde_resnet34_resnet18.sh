python datafree_kd.py \
--method pretrained \
--pretrained_mode sde \
--pretrained_G_weight /data1/lijingru/score_sde_pytorch/exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth \
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
--seed 0 \
--log_tag pretrained_sde \
--curr_option none \
--data_root /data/lijingru/cifar10/ 