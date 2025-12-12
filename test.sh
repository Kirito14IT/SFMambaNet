cd core/

python main.py --run_mode=test      \
              --model_path=../log/train/ \
              --res_path=../log/yfcc_know \
              --data_te=/root/autodl-tmp/data_dump/yfcc-sift-2000-testknown.hdf5 \
              --use_ransac=True \
              --log_base=../log/
python main.py --run_mode=test      \
              --model_path=../log/train/ \
              --res_path=../log/yfcc_know \
              --data_te=/root/autodl-tmp/data_dump/yfcc-sift-2000-testknown.hdf5 \
              --use_ransac=False \
              --log_base=../log/
python main.py --run_mode=test      \
              --model_path=../log/train/ \
              --res_path=../log/yfcc_unknow \
              --data_te=/root/autodl-tmp/data_dump/yfcc-sift-2000-test.hdf5 \
              --use_ransac=True \
              --log_base=../log/

# 首先运行这个测试
python main.py --run_mode=test      \
              --model_path=../log/train/ \
              --res_path=../log/yfcc_unknow \
              --data_te=/root/autodl-tmp/data_dump/yfcc-sift-2000-test.hdf5 \
              --use_ransac=False \
              --log_base=../log/