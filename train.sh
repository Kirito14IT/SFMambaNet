
##### yfcc-sift-2000
cd core/
python main.py --run_mode=train      \
              --log_base=../log_lsgax1_cdcpx6_v7_v1/     \
              --train_iter=500000 \
              --data_tr=/root/autodl-tmp/data_dump/yfcc-sift-2000-train.hdf5 \
              --data_va=/root/autodl-tmp/data_dump/yfcc-sift-2000-val.hdf5

# 查看tensorboard
tensorboard --logdir=/root/tf-logs/ --port=6009
