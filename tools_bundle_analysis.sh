# Bundle Training Analysis
CONFIG=$1
JSON=$2
INPUT_SHAPE=${INPUT_SHAPE:-[3,2048,1024]}
LOSS_IMAGE=${LOSS_IMAGE:-losses.png}
MAP_IMAGE=${MAP_IMAGE:-mAPs.png}

# # Model Complexity
# python tools/analysis_tools/get_flops.py $CONFIG --gpu_id 7

# Avg Training speed
python tools/results_analysis.py cal_train_time $JSON

# Plots
## Loss
python tools/results_analysis.py plot_curve $JSON --keys loss_cls loss_bbox loss_mask loss --out $LOSS_IMAGE --title $CONFIG --font_size 21

## mAP
python tools/results_analysis.py plot_curve $JSON --keys segm_mAP bbox_mAP --out $MAP_IMAGE --title $CONFIG --font_size 21
# python tools/analysis_tools/analyze_logs.py plot_curve $JSON --keys segm_mAP bbox_mAP --out mAP_sns.png