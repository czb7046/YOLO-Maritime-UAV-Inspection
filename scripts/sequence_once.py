import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import random
import os

# ==========================================
# 1. 配置参数
# ==========================================
EXCEL_CACHE_PATH = 'sequence_once_predictions_cache.xlsx'  
LOG_DIR = 'flight_logs'            
os.makedirs(LOG_DIR, exist_ok=True)

MONTE_CARLO_TRIALS = 10000      
MAX_STEPS = 3000               

# 假设视频是顺时针绕船 (尾->左->首->右->尾)
VIDEO_IS_CLOCKWISE = True 

CLASS_MAP = {0: 'Stern', 1: 'Bow', 2: 'Left', 3: 'Right', 4: 'Top', -1: 'None'}

# ==========================================
# 2. 核心仿真类
# ==========================================
class UAVFinalEvaluator:
    def __init__(self):
        self.raw_preds = []         # YOLO 原始预测
        self.true_labels = []       # 你手工修正的真值
        self.total_frames = 0
        self.target_zone = set()    # 真正的 Stern 区域

    def load_data(self):
        print(f"[*] 加载数据 {EXCEL_CACHE_PATH} ...")
        df = pd.read_excel(EXCEL_CACHE_PATH)
        
        self.raw_preds = df['Class_ID'].fillna(-1).astype(int).tolist()
        
        # 处理 Corrected_class
        mapping = {'stern': 0, 'bow': 1, 'left': 2, 'right': 3, 'top': 4}
        def parse_corrected(val):
            if pd.isna(val) or str(val).strip() == '': return -1
            if isinstance(val, (int, float)): return int(val)
            return mapping.get(str(val).strip().lower(), -1)

        corrected = [parse_corrected(x) for x in df['Corrected_class']]
        
        # 如果没有人工修正，则默认等于 raw
        for i in range(len(corrected)):
            if corrected[i] == -1:
                corrected[i] = self.raw_preds[i]
                
        self.true_labels = corrected
        self.total_frames = len(self.raw_preds)
        
        # 目标区域必须是物理上真实的 Stern
        self.target_zone = {i for i, p in enumerate(self.true_labels) if p == 0}
        print(f"[*] 数据加载完毕。总帧数: {self.total_frames}, 物理目标(Stern)区间大小: {len(self.target_zone)}帧")

    def _get_direction(self, perception_class):
        """核心方向计算逻辑"""
        if perception_class == -1 or perception_class == 4: return 0 
        if VIDEO_IS_CLOCKWISE:
            if perception_class == 2: return -1  # 左舷 -> 倒退找船尾
            if perception_class == 3: return 1   # 右舷 -> 快进找船尾
            if perception_class == 1: return random.choice([-1, 1]) 
        else:
            if perception_class == 2: return 1   
            if perception_class == 3: return -1 
            if perception_class == 1: return random.choice([-1, 1])
        return 0

    # def simulate_bss(self, start_frame):
    #     """1. BSS 盲搜：开局随机猜方向，一路走到黑"""
    #     current = start_frame
    #     direction = random.choice([-1, 1]) 
    #     steps = 0
    #     while current not in self.target_zone and steps < MAX_STEPS:
    #         current = (current + direction) % self.total_frames 
    #         steps += 1
    #     return steps

    # def simulate_initial_gss(self, start_frame):
    #     """2. Initial-GSS (你的策略)：只在起点用 YOLO 预测方向，然后一路走到黑"""
    #     current = start_frame
    #     # 使用原始 YOLO 预测来决定初始方向
    #     initial_perception = self.raw_preds[current]
    #     direction = self._get_direction(initial_perception)
        
    #     # 如果起点刚好没认出来，随机猜一个
    #     if direction == 0: direction = random.choice([-1, 1]) 
            
    #     steps = 0
    #     while current not in self.target_zone and steps < MAX_STEPS:
    #         current = (current + direction) % self.total_frames
    #         steps += 1
    #     return steps

    # def simulate_ideal_gss(self, start_frame):
    #     """3. Ideal-GSS (理论上限)：在起点用完美的 真值 预测方向，然后走到黑"""
    #     current = start_frame
    #     # 使用你手工修正的完美标签决定方向
    #     true_perception = self.true_labels[current]
    #     direction = self._get_direction(true_perception)
    #     if direction == 0: direction = random.choice([-1, 1]) 
            
    #     steps = 0
    #     while current not in self.target_zone and steps < MAX_STEPS:
    #         current = (current + direction) % self.total_frames
    #         steps += 1
    #     return steps
    def simulate_bss(self, start_frame):
        current = start_frame
        direction = random.choice([-1, 1]) 
        steps = 0
        while current not in self.target_zone and steps < MAX_STEPS:
            current = (current + direction) % self.total_frames 
            steps += 1
        return steps, current  # <--- 修改点：同时返回步数和结束时的帧索引

    def simulate_initial_gss(self, start_frame):
        current = start_frame
        initial_perception = self.raw_preds[current]
        direction = self._get_direction(initial_perception)
        
        if direction == 0: direction = random.choice([-1, 1]) 
            
        steps = 0
        while current not in self.target_zone and steps < MAX_STEPS:
            current = (current + direction) % self.total_frames
            steps += 1
        return steps, current  # <--- 修改点

    def simulate_ideal_gss(self, start_frame):
        current = start_frame
        true_perception = self.true_labels[current]
        direction = self._get_direction(true_perception)
        
        if direction == 0: direction = random.choice([-1, 1]) 
            
        steps = 0
        while current not in self.target_zone and steps < MAX_STEPS:
            current = (current + direction) % self.total_frames
            steps += 1
        return steps, current  # <--- 修改点

    def simulate_continuous_gss_with_log(self, start_frame, trial_id):
        """4. 闭环 GSS (证明它为什么会失败，带日志)"""
        current = start_frame
        steps = 0
        trajectory = []
        
        while current not in self.target_zone and steps < MAX_STEPS:
            perception = self.raw_preds[current]
            direction = self._get_direction(perception)
            if direction == 0: direction = random.choice([-1, 1]) 
                
            trajectory.append({
                'Step': steps, 'Frame': current, 
                'YOLO_Saw': CLASS_MAP.get(perception, 'Unknown'), 
                'True_Class': CLASS_MAP.get(self.true_labels[current], 'Unknown'),
                'Move': direction
            })
            
            current = (current + direction) % self.total_frames
            steps += 1
            
        # 如果陷入震荡 (比如超过 1500 步)，保存日志供你查阅
        if steps >= 1500:
            df_log = pd.DataFrame(trajectory)
            df_log.to_csv(f"{LOG_DIR}/oscillation_trial_{trial_id}_start_{start_frame}.csv", index=False)
            
        return steps
    # 在 sequence_eval.py 的 UAVFinalEvaluator 类中修改或添加这个方法：
    # def run_simulation(self, trials=10000):
    #     """只跑仿真并返回纯数据，不画图"""
    #     res_bss, res_initial_gss, res_ideal_gss = [], [], []
        
    #     for _ in range(trials):
    #         # 随机选取一个非目标区域的起点
    #         start = random.randint(0, self.total_frames - 1)
    #         while start in self.target_zone:
    #             start = random.randint(0, self.total_frames - 1)
                
    #         res_bss.append(self.simulate_bss(start))
    #         res_initial_gss.append(self.simulate_initial_gss(start))
    #         res_ideal_gss.append(self.simulate_ideal_gss(start))
            
    #     return res_bss, res_initial_gss, res_ideal_gss
    # 请在 UAVFinalEvaluator 类中替换这个方法
    def run_simulation(self, trials=10000):
        """只跑仿真并返回详细数据，用于宏观统计和日志记录"""
        
        # 建立字典来存储带日志的完整数据
        res_bss = {'steps': [], 'logs': []}
        res_ini = {'steps': [], 'logs': []}
        res_ideal = {'steps': [], 'logs': []}
        
        for _ in range(trials):
            start = random.randint(0, self.total_frames - 1)
            while start in self.target_zone:
                start = random.randint(0, self.total_frames - 1)
                
            # 运行 BSS
            steps_b, end_b = self.simulate_bss(start)
            res_bss['steps'].append(steps_b)
            res_bss['logs'].append((start, end_b, steps_b))
            
            # 运行 Initial-GSS
            steps_i, end_i = self.simulate_initial_gss(start)
            res_ini['steps'].append(steps_i)
            res_ini['logs'].append((start, end_i, steps_i))
            
            # 运行 Ideal-GSS
            steps_id, end_id = self.simulate_ideal_gss(start)
            res_ideal['steps'].append(steps_id)
            res_ideal['logs'].append((start, end_id, steps_id))
            
        return res_bss, res_ini, res_ideal
        
    def run(self):
        print(f"[*] 开始蒙特卡洛仿真 ({MONTE_CARLO_TRIALS} 次) ...")
        res_bss, res_initial_gss, res_ideal_gss, res_continuous = [], [], [], []
        
        for trial in range(MONTE_CARLO_TRIALS):
            start = random.randint(0, self.total_frames - 1)
            while start in self.target_zone:
                start = random.randint(0, self.total_frames - 1)
                
            res_bss.append(self.simulate_bss(start))
            res_initial_gss.append(self.simulate_initial_gss(start))
            res_ideal_gss.append(self.simulate_ideal_gss(start))
            
            # 跑一次连续闭环，仅仅为了收集失败日志
            if trial < 50: # 没必要跑 1000 次，跑 50 次足够抓出异常日志了
                res_continuous.append(self.simulate_continuous_gss_with_log(start, trial))
            
        self.plot_results(res_bss, res_initial_gss, res_ideal_gss)

    def plot_results(self, bss, initial_gss, ideal_gss):
        plt.figure(figsize=(10, 6))
        
        # 只取没有死循环的数据绘图
        # bss_clean = [x for x in bss if x < MAX_STEPS]
        # ini_clean = [x for x in initial_gss if x < MAX_STEPS]
        # ideal_clean = [x for x in ideal_gss if x < MAX_STEPS]
        bss_clean = [x for x in bss]
        ini_clean = [x for x in initial_gss]
        ideal_clean = [x for x in ideal_gss]

        box = plt.boxplot([bss_clean, ini_clean, ideal_clean], 
                          labels=['Blind Search\n(BSS)', 'Initial-GSS\n(YOLO Driven)', 'Ideal-GSS\n(Perfect Perception)'], 
                          patch_artist=True, notch=True)
                          
        colors = ['#FF9999', '#66B2FF', '#99FF99']
        for patch, color in zip(box['boxes'], colors): patch.set_facecolor(color)
            
        plt.title('Sequence-Level Evaluation: Search Strategy Efficiency', fontsize=14)
        plt.ylabel('Frames Required to Reach Target', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        bss_mean = np.mean(bss_clean)
        ini_mean = np.mean(ini_clean)
        ideal_mean = np.mean(ideal_clean)
        
        gain_ini = (bss_mean - ini_mean) / bss_mean * 100
        gain_ideal = (bss_mean - ideal_mean) / bss_mean * 100
        
        textstr = '\n'.join((
            r'$\mu_{BSS}=%.1f$' % bss_mean,
            r'$\mu_{Initial-GSS}=%.1f$ (Gain: %.1f%%)' % (ini_mean, gain_ini),
            r'$\mu_{Ideal-GSS}=%.1f$ (Gain: %.1f%%)' % (ideal_mean, gain_ideal)))
            
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
                
        plt.savefig('final_strategy_evaluation.png', dpi=300, bbox_inches='tight')
        plt.savefig('final_strategy_evaluation.pdf', dpi=300, bbox_inches='tight')
        print(f"\n========== 最终量化结论 ==========")
        print(textstr)
        print("[*] 图表已保存为 final_strategy_evaluation.png")
        print(f"[*] 闭环失败日志(若有)已保存至 {LOG_DIR} 文件夹。")
        plt.show()

if __name__ == "__main__":
    evaluator = UAVFinalEvaluator()
    evaluator.load_data()
    evaluator.run()
