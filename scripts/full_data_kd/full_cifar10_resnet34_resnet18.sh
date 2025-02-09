python vanilla_kd.py \
--data_root /data/lijingru/cifar10/ \
--teacher resnet34 \
--student resnet18 \
--dataset cifar10 \
--lr 0.01 \
--T 20 \
--epochs 200 \
--batch_size 512 \
--gpu 1 \
--curr_option none \
--lambda_0 3 \
--log_tag vanilla_kd