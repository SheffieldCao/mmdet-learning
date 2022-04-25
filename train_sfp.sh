PORT=29502 CUDA_VISIBLE_DEVICES=2,3 bash tools/dist_train.sh configs/mask_rcnn/mask_rcnn_gpwin8_sfp_ws_gn_cs.py 2 \
            --work-dir 'outputs/test_gpwin_sfp_bot_up' \
            --deterministic