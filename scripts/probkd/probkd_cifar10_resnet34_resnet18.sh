python datafree_kd.py \
--method probkd \
--dataset cifar10 \
--batch_size 768 \
--teacher resnet34 \
--student resnet18 \
--lr 0.05 \
--epochs 250 \
--kd_steps 5 \
--kd_steps_interval 10 \
--g_steps_interval 1 \
--ep_steps 400 \
--g_steps 1 \
--lr_g 0.001 \
--begin_fraction 0.25 \
--end_fraction 0.75 \
--grad_adv 0.5 \
--adv 0. \
--depth 2 \
--T 5 \
--lmda_ent -20 \
--oh 1 \
--act 0. \
--gpu 2 \
--seed 0 \
--bn 1 \
--save_dir run/probkd_test \
--log_tag probkd_L2_line93 \
--data_root /data/lijingru/cifar10/ \
--no_feature \
--adv_type kl \
--curr_option curr_log \
--lambda_0 2.0 \
--loss kl
# --resume /data/lijingru/DataFree/checkpoints/datafree-probkd/cifar10-resnet34-resnet18--probkd_dcgan_L2_adv_testrelu2.pth
