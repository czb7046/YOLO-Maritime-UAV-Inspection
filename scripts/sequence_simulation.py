import numpy as np
import pandas as pd
import os
from sequence_once import UAVFinalEvaluator  # 确保这里导入的名字和你实际的文件名一致

# ==========================================
# 1. 宏观统计配置
# ==========================================
MACRO_RUNS = 5              
TRIALS_PER_RUN = 10000      
VIDEO_TOTAL_FRAMES = 1114   
FAILURE_THRESHOLD = VIDEO_TOTAL_FRAMES / 2  

LOGS_DIR = 'macro_ruuner_logs'  # <--- 新增：统一的日志保存文件夹

# ==========================================
# 2. 自动化执行与数据收集
# ==========================================
def save_logs_to_csv(run_idx, bss_data, ini_data, ideal_data):
    """辅助函数：将一次 Macro-run 的详细数据保存为 CSV"""
    
    # 构建 DataFrame 结构
    # 格式要求: [Runs, Start_frame, End_frame, Total_frames(这里指耗时的 steps)]
    def format_log_data(log_list, strategy_name):
        formatted = []
        for start_f, end_f, steps in log_list:
            formatted.append({
                'Runs': run_idx + 1,
                'Strategy': strategy_name,
                'Start_frame': start_f,
                'End_frame': end_f,
                'Total_frames': steps  # 耗时的步数
            })
        return formatted

    all_logs = []
    all_logs.extend(format_log_data(bss_data['logs'], 'BSS'))
    all_logs.extend(format_log_data(ini_data['logs'], 'Initial-GSS'))
    all_logs.extend(format_log_data(ideal_data['logs'], 'Ideal-GSS'))

    df = pd.DataFrame(all_logs)
    
    # 生成文件名并保存
    csv_filename = os.path.join(LOGS_DIR, f"macro_run_{run_idx + 1}_logs.csv")
    df.to_csv(csv_filename, index=False)
    print(f"    [+] Run {run_idx + 1} 详细日志已保存至: {csv_filename}")


def run_macro_statistics():
    # 确保日志文件夹存在
    os.makedirs(LOGS_DIR, exist_ok=True)
    
    print(f"[*] 开始宏观统计测试: {MACRO_RUNS} 次独立运行，每次 {TRIALS_PER_RUN} Trials ...\n")
    print(f"[*] 所有仿真轨迹日志将保存在 ./{LOGS_DIR}/ 目录下。\n")
    
    evaluator = UAVFinalEvaluator()
    evaluator.load_data()  
    
    macro_bss_means = []
    macro_ini_means = []
    macro_ideal_means = []
    macro_ini_success_rates = []

    for i in range(MACRO_RUNS):
        print(f"--- 正在执行 Macro-run {i+1}/{MACRO_RUNS} ...")
        
        # 接收带有详细日志的字典数据
        res_bss, res_ini, res_ideal = evaluator.run_simulation(trials=TRIALS_PER_RUN)
        
        # 保存这 10000 次 trial 的详细轨迹到 CSV
        save_logs_to_csv(i, res_bss, res_ini, res_ideal)
        
        # 1. 记录平均帧数 (从 'steps' 列表中取)
        mean_bss = np.mean(res_bss['steps'])
        mean_ini = np.mean(res_ini['steps'])
        mean_ideal = np.mean(res_ideal['steps'])
        
        macro_bss_means.append(mean_bss)
        macro_ini_means.append(mean_ini)
        macro_ideal_means.append(mean_ideal)
        
        # 2. 计算 Empirical SSR
        success_count = sum(1 for steps in res_ini['steps'] if steps < FAILURE_THRESHOLD)
        empirical_ssr = (success_count / TRIALS_PER_RUN) * 100
        macro_ini_success_rates.append(empirical_ssr)

    # ==========================================
    # 3. 计算最终的统计学指标
    # ==========================================
    final_bss_mean, final_bss_std = np.mean(macro_bss_means), np.std(macro_bss_means)
    final_ini_mean, final_ini_std = np.mean(macro_ini_means), np.std(macro_ini_means)
    final_ideal_mean, final_ideal_std = np.mean(macro_ideal_means), np.std(macro_ideal_means)
    
    gains_ini = [((b - i) / b) * 100 for b, i in zip(macro_bss_means, macro_ini_means)]
    final_gain_ini_mean, final_gain_ini_std = np.mean(gains_ini), np.std(gains_ini)
    
    gains_ideal = [((b - i) / b) * 100 for b, i in zip(macro_bss_means, macro_ideal_means)]
    final_gain_ideal_mean, final_gain_ideal_std = np.mean(gains_ideal), np.std(gains_ideal)
    
    final_ssr_mean, final_ssr_std = np.mean(macro_ini_success_rates), np.std(macro_ini_success_rates)

    # ==========================================
    # 4. 打印学术报告
    # ==========================================
    print("\n" + "="*50)
    print(" 🚀 宏观统计结果 (基于 5 x 10,000 Trials)")
    print("="*50)
    print(f"[1] Time-to-Target (TtT) 平均寻的帧数:")
    print(f"    - Blind Search (BSS): {final_bss_mean:.1f} ± {final_bss_std:.1f} frames")
    print(f"    - Initial-GSS (YOLO): {final_ini_mean:.1f} ± {final_ini_std:.1f} frames")
    print(f"    - Ideal-GSS (Truth) : {final_ideal_mean:.1f} ± {final_ideal_std:.1f} frames")
    
    print(f"\n[2] Efficiency Gain (续航节省比例 vs. BSS):")
    print(f"    - Initial-GSS 真实增益: {final_gain_ini_mean:.1f}% ± {final_gain_ini_std:.1f}%")
    print(f"    - Ideal-GSS 理论上限: {final_gain_ideal_mean:.1f}% ± {final_gain_ideal_std:.1f}%")
    
    print(f"\n[3] 验证公式 SSR ≈ Accuracy (异常点分析核心数据):")
    print(f"    - Initial-GSS 经验战略成功率 (Empirical SSR): {final_ssr_mean:.1f}% ± {final_ssr_std:.1f}%")
    print("="*50)

if __name__ == "__main__":
    run_macro_statistics()