cd ~/rknn-train/cw/rk3588-pytorch
make -C csrc 2>&1 | grep -E "error|warning" | head
for t in test_wgrad test_conv test_conv_out16; do echo "## $t"; python tests/$t.py 2>&1 | grep -E "FAIL|ALL PASS|FAILURES|passed|Error" | head -5; done
python tests/test_zoo.py --only resnet18,unet,mobilenet --reps 2 2>&1 | grep -E "^(resnet|unet|mobile|all)"
rm -f tests/_tmp_*.sh
