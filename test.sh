SHOTS=$1
DATA_PATH=$2
TEST_ITERS=1000
FT_ITERS=800

python fs_test_cli.py \
--root_data_path $DATA_PATH \
--shots $SHOTS \
--model_type CONV4 \
--model_path "./weights/CONV4.pth" \
--img_size 224 \
--test_iters $TEST_ITERS \
--ft_iters $FT_ITERS

python fs_test_cli.py \
--root_data_path $DATA_PATH \
--shots $SHOTS \
--model_type RESNET18 \
--model_path "./weights/RESNET18.pth" \
--img_size 224 \
--test_iters $TEST_ITERS \
--ft_iters $FT_ITERS

python fs_test_cli.py \
--root_data_path $DATA_PATH \
--shots $SHOTS \
--model_type DINO_SMALL \
--model_path "./weights/DINO.pth" \
--img_size 224 \
--test_iters $TEST_ITERS \
--ft_iters $FT_ITERS