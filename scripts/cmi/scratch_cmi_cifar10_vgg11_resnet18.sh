python datafree_kd.py \
--method cmi \
--dataset cifar10 \
--batch_size 256 \
--synthesis_batch_size 512 \
--teacher vgg11 \
--student resnet18 \
--lr 0.1 \
--kd_steps 400 \
--ep_steps 400 \
--g_steps 200 \
--lr_g 1e-3 \
--adv 0.5 \
--bn 1.0 \
--oh 1.0 \
--cr 0.8 \
--cr_T 0.1 \
--act 0 \
--balance 0 \
--gpu 0 \
--seed 0 \
--T 20 \
--save_dir run/scratch_cmi_b \
--curr_option none \
--data_root ~/cifar10 \
--log_fidelity \
--log_tag scratch_cmi_b