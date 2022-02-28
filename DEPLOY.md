# Run
## new
    python3 i_infer.py --images_path='/home/zhuhao/dataset/tmp/images'

## old
    python3 q2l_infer.py -a 'Q2L-CvT_w24-384' --config "/home/zhuhao/myModel/query2labels/Q2L-CvT_w24-384/config_new.json" -b 8 --resume '/home/zhuhao/myModel/query2labels/Q2L-CvT_w24-384/checkpoint.pkl' --dataset_dir '/home/zhuhao/dataset/public/coco'