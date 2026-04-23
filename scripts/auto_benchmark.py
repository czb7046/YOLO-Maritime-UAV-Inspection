import os
import time
import subprocess
import smtplib
import shutil  # 【新增】用于移动文件
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np
import cv2
from ultralytics import YOLO

# ================= 配置区 =================
MODELS_DIR = "./models"  # 存放 25个 pt 模型的文件夹
DATA_YAML = "./my_data.yaml" # 数据集配置
VIDEO_PATH = "./testvideo/ship.mp4" # 用于测速和功耗的绕船视频
OUTPUT_CSV = "benchmark_results_final.csv"
LOGS_BASE_DIR = "./logs"  # 【新增】存放所有原始功耗日志的主目录

# 邮件配置
SMTP_SERVER = "smtp.126.com"
SMTP_PORT = 465
SENDER_EMAIL = "caozhibo@126.com"  
SENDER_PASSWORD = "**************" 
RECEIVER_EMAIL = "caozhibo@126.com" 

NUM_RUNS = 5 # FPS和功耗测5次求方差
TEST_DURATION = 60 # 每次满载测试持续时间（秒）。60秒 x 5次 = 每个模型跑5分钟
# ==========================================

# 全局变量，用于存放预加载的帧
PRELOADED_FRAMES = []

def load_frames_to_memory():
    """将视频帧预先加载到内存中，彻底消除解码和磁盘IO带来的测速瓶颈"""
    global PRELOADED_FRAMES
    print(f"📦 Pre-loading frames from {VIDEO_PATH} into memory...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    while len(PRELOADED_FRAMES) < 100:  # 缓存 100 帧即可，反复循环推理
        ret, frame = cap.read()
        if not ret: break
        # 将画面缩放为接近模型输入的尺寸，减少预处理开销
        frame = cv2.resize(frame, (320, 320))
        PRELOADED_FRAMES.append(frame)
    cap.release()
    print(f"✅ Loaded {len(PRELOADED_FRAMES)} frames into RAM.")

def send_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = SENDER_EMAIL
        msg['To'] = RECEIVER_EMAIL
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print(f"📧 Email sent: {subject}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def measure_power_and_fps(model_path, model_name, model_type, runs=5):
    """通用的测速与测功耗函数，支持 .pt 和 .engine"""
    fps_list = []
    power_list = []
    
    # 【新增】创建目标日志文件夹，例如： ./logs/yolov10l_pt/
    target_log_dir = os.path.join(LOGS_BASE_DIR, f"{model_name}_{model_type}")
    os.makedirs(target_log_dir, exist_ok=True)

    model = YOLO(model_path, task='detect')
    
    print("      Warming up GPU...")
    for i in range(10):
        _ = model.predict(source=PRELOADED_FRAMES[0], imgsz=320, device=0, verbose=False)
        
    for r in range(runs):
        print(f"      Run {r+1}/{runs} for Power/FPS (Duration: {TEST_DURATION}s)...")
        log_file = f"tegrastats_log_{r}.txt"
        
        tegra_proc = subprocess.Popen(['tegrastats', '--interval', '500', '--logfile', log_file])
        
        # 【核心修改】：强制运行指定的时间（例如 60 秒），而不是固定帧数
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < TEST_DURATION:
            frame = PRELOADED_FRAMES[frame_count % len(PRELOADED_FRAMES)]
            _ = model.predict(source=frame, imgsz=320, device=0, verbose=False)
            frame_count += 1
            
        end_time = time.time()
        
        tegra_proc.terminate()
        tegra_proc.wait()
        
        # 计算该段时间内的平均 FPS
        fps = frame_count / (end_time - start_time)
        fps_list.append(fps)
        
        # 解析 tegrastats 日志提取 VDD_IN 功耗
        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    run_powers = []
                    for line in lines:
                        if 'VDD_IN' in line:
                            power_str = line.split('VDD_IN ')[1].split('mW/')[0].strip()
                            power_mw = int(power_str)
                            run_powers.append(power_mw / 1000.0) # 转为 W
                    if len(run_powers) > 10:
                        # 抛弃前2秒(前4个点)的瞬态功耗，取稳态均值
                        power_list.append(np.mean(run_powers[4:]))
                    elif run_powers:
                        power_list.append(np.mean(run_powers))
                    else:
                        power_list.append(0.0)
        except Exception as e:
            print(f"      Error parsing tegrastats: {e}")
            power_list.append(0.0)
            
         # 【修改】不再使用 os.remove(log_file)，而是移动到专属文件夹
        try:
            if os.path.exists(log_file):
                final_log_path = os.path.join(target_log_dir, f"run_{r+1}.txt")
                shutil.move(log_file, final_log_path)
        except Exception as e:
            print(f"      Error moving log file: {e}")

        time.sleep(5) # 【修改】两次烤机之间休息5秒钟，让芯片稍微散热
        
    return np.mean(fps_list), np.std(fps_list), np.mean(power_list), np.std(power_list)

def main():
    # 程序开始时，加载视频到内存
    load_frames_to_memory()
    if not PRELOADED_FRAMES:
        print("❌ Error: Failed to load video frames. Check VIDEO_PATH.")
        return

    pt_models = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pt')]
    all_results = []
    
    send_email("🚀 Benchmark Started", f"Found {len(pt_models)} models. Starting PT and Engine benchmark on Jetson.")
    
    for idx, pt_file in enumerate(pt_models):
        model_name = pt_file.replace('.pt', '')
        pt_path = os.path.join(MODELS_DIR, pt_file)
        engine_path = pt_path.replace('.pt', '.engine')
        
        result_dict = {'Model': model_name}
        print(f"\n[{idx+1}/{len(pt_models)}] Processing {model_name}...")
        
        try:
            # ================= 1. 测试 PT 模型 =================
            print(" -> [PT] Evaluating mAP...")
            model_pt = YOLO(pt_path)
            metrics_pt = model_pt.val(data=DATA_YAML, imgsz=320, batch=1, device=0, verbose=False)
            result_dict['PT_mAP50'] = metrics_pt.box.map50
            
            classes = metrics_pt.names
            for i, c in enumerate(metrics_pt.box.ap_class_index):
                result_dict[f'PT_{classes[c]}_mAP50'] = metrics_pt.box.maps[i]
                
            print(" -> [PT] Measuring FPS and Power (5 runs)...")
            pt_fps_m, pt_fps_s, pt_pwr_m, pt_pwr_s = measure_power_and_fps(pt_path, model_name, 'pt', runs=NUM_RUNS)
            result_dict['PT_FPS_Mean'] = pt_fps_m
            result_dict['PT_FPS_Std'] = pt_fps_s
            result_dict['PT_Power_W_Mean'] = pt_pwr_m
            result_dict['PT_Power_W_Std'] = pt_pwr_s
            
            # ================= 2. 导出 TensorRT =================
            if not os.path.exists(engine_path):
                print(" -> Exporting to TensorRT Engine...")
                model_pt.export(format='engine', imgsz=320, half=True, workspace=4, dynamic=False, device=0)
            
            # ================= 3. 测试 Engine 模型 =================
            print(" -> [Engine] Evaluating mAP...")
            model_engine = YOLO(engine_path, task='detect')
            metrics_engine = model_engine.val(data=DATA_YAML, imgsz=320, batch=1, device=0, verbose=False)
            result_dict['Engine_mAP50'] = metrics_engine.box.map50
            
            for i, c in enumerate(metrics_engine.box.ap_class_index):
                result_dict[f'Engine_{classes[c]}_mAP50'] = metrics_engine.box.maps[i]
            
            print(" -> [Engine] Measuring FPS and Power (5 runs)...")
            eng_fps_m, eng_fps_s, eng_pwr_m, eng_pwr_s = measure_power_and_fps(engine_path, model_name, 'engine', runs=NUM_RUNS)
            result_dict['Engine_FPS_Mean'] = eng_fps_m
            result_dict['Engine_FPS_Std'] = eng_fps_s
            result_dict['Engine_Power_W_Mean'] = eng_pwr_m
            result_dict['Engine_Power_W_Std'] = eng_pwr_s
            
            # ================= 保存与记录 =================
            all_results.append(result_dict)
            pd.DataFrame(all_results).to_csv(OUTPUT_CSV, index=False)
            print(f" -> Completed {model_name}.")
            send_email(f"✅ {model_name} Completed", f"{model_name} processed successfully. Its result saved to {OUTPUT_CSV}.")

        except Exception as e:
            error_msg = f"❌ Error processing {model_name}: {str(e)}"
            print(error_msg)
            send_email(f"⚠️ Benchmark Error: {model_name}", error_msg)
            continue
            
    summary = f"All {len(pt_models)} models processed successfully. Results saved to {OUTPUT_CSV}."
    send_email("✅ Benchmark Completed", summary)
    print("\n🎉 All Done! " + summary)

if __name__ == "__main__":
    main()