cd core/

python main.py --run_mode=test      \
              --model_path=../log/train/ \
              --res_path=../log/yfcc_know \
              --data_te=/home/lab603/Documents/OADATA/yfcc-sift-2000-testknown.hdf5 \
              --use_ransac=True \
              --log_base=../log/
python main.py --run_mode=test      \
              --model_path=../log/train/ \
              --res_path=../log/yfcc_know \
              --data_te=/home/lab603/Documents/OADATA/yfcc-sift-2000-testknown.hdf5 \
              --use_ransac=False \
              --log_base=../log/
python main.py --run_mode=test      \
              --model_path=../log/train/ \
              --res_path=../log/yfcc_unknow \
              --data_te=/home/lab603/Documents/OADATA/yfcc-sift-2000-test.hdf5 \
              --use_ransac=True \
              --log_base=../log/
python main.py --run_mode=test      \
              --model_path=../log_lsgax1_cdcpx6_v7_v1/train/ \
              --res_path=../log_lsgax1_cdcpx6_v7_v1/yfcc_unknow \
              --data_te=/root/autodl-tmp/data_dump/yfcc-sift-2000-test.hdf5 \
              --use_ransac=False \
              --log_base=../log_lsgax1_cdcpx6_v7_v1/