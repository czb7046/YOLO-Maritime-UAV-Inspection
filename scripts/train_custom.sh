#!/bin/bash

# ====================== 你的配置 ======================
DATA_YAML="my_data.yaml"
IMGSZ=320
EPOCHS=80
FLIPLR=0
PROJECT_ROOT="./works"
BATCH=32
DEVICE="0,1"          # 双卡4090D
WEIGHTS_DIR="./yolo_weights"
LOG_DIR="./training_logs"
# ======================================================

mkdir -p "$WEIGHTS_DIR"
mkdir -p "$LOG_DIR"

MODEL_LIST=(
    yolov8n.pt yolov8s.pt yolov8m.pt yolov8l.pt yolov8x.pt
    yolov9t.pt yolov9s.pt yolov9m.pt yolov9c.pt yolov9e.pt
    yolov10n.pt yolov10s.pt yolov10m.pt yolov10l.pt yolov10x.pt
    yolo11n.pt yolo11s.pt yolo11m.pt yolo11l.pt yolo11x.pt
    yolo12n.pt yolo12s.pt yolo12m.pt yolo12l.pt yolo12x.pt
)


for MODEL in "${MODEL_LIST[@]}"; do
    MODEL_NAME=${MODEL%.pt}
    LOCAL_FILE="${WEIGHTS_DIR}/${MODEL}"
    LOG_FILE="${LOG_DIR}/${MODEL_NAME}_train.log"

    echo -e "\n=================================================="
    echo "🚀 训练模型：$MODEL_NAME"
    echo "=================================================="

    # 智能加载：本地有就用本地，没有就自动下载
    if [ -f "$LOCAL_FILE" ]; then
        echo "✅ 使用本地模型：$LOCAL_FILE"
        USE_MODEL="$LOCAL_FILE"
    else
        echo "⚠️ 本地无模型，自动从官方下载：$MODEL"
        USE_MODEL="$MODEL"
    fi

    # ==================== 核心修复 ====================
    # 去掉 DDP 多进程，改用 ultralytics 自带稳定双卡模式
    # 不会再报 importlib_metadata 错误！
    yolo detect train \
        model="$USE_MODEL" \
        data="$DATA_YAML" \
        imgsz="$IMGSZ" \
        epochs="$EPOCHS" \
        fliplr="$FLIPLR" \
        project="$PROJECT_ROOT" \
        name="$MODEL_NAME" \
        batch=$BATCH \
        device=$DEVICE \
        exist_ok=True 2>&1 | tee "$LOG_FILE"

    # 导出 engine
    BEST_PT="./runs/detect/${PROJECT_ROOT}/${MODEL_NAME}/weights/best.pt"
    yolo export model="$BEST_PT" format=engine imgsz=$IMGSZ device=$DEVICE half=True workspace=4 2>&1 | tee -a "$LOG_FILE"

    echo -e "\n🎉 $MODEL_NAME 完成！"
done

echo -e "🎉🎉🎉 所有模型训练完成！"
