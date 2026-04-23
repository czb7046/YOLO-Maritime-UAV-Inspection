import pandas as pd
import numpy as np

# 1. 配置文件路径
INPUT_CSV = 'benchmark_results_final_1st.csv' # 请确保该文件与脚本在同一目录下
OUTPUT_EXCEL = 'WNS_Verification.xlsx'

# 2. 读取 CSV 数据
print(f"Reading data from {INPUT_CSV}...")
df = pd.read_csv(INPUT_CSV)

# 3. 将单行数据展开为两行 (一行 PT，一行 Engine)
# 这样方便找到全局的 mAP_max 和 mAP_min，也符合我们在表格中的展示方式
records = []
for index, row in df.iterrows():
    # 提取 PT 模型数据
    records.append({
        'Model_Name': f"{row['Model']}-pt",
        'mAP50': row['PT_mAP50'],
        'FPS_Mean': row['PT_FPS_Mean'],
        'FPS_Std': row['PT_FPS_Std'],
        'Power_Mean': row['PT_Power_W_Mean'],
        'Power_Std': row['PT_Power_W_Std']
    })
    # 提取 Engine 模型数据
    records.append({
        'Model_Name': f"{row['Model']}-engine",
        'mAP50': row['Engine_mAP50'],
        'FPS_Mean': row['Engine_FPS_Mean'],
        'FPS_Std': row['Engine_FPS_Std'],
        'Power_Mean': row['Engine_Power_W_Mean'],
        'Power_Std': row['Engine_Power_W_Std']
    })

# 转换为 DataFrame
df_calc = pd.DataFrame(records)

# 4. 计算全局 mAP 的最值，用于归一化
# MAP_MIN = df_calc['mAP50'].min()
# MAP_MAX = df_calc['mAP50'].max()
MAP_MIN = 0
MAP_MAX = 1
print(f"Global mAP50 range for normalization: Min = {MAP_MIN:.4f}, Max = {MAP_MAX:.4f}")

# 5. 定义 WNS 计算函数
def calc_norm_map(m):
    return (m - MAP_MIN) / (MAP_MAX - MAP_MIN)

def calc_norm_fps(f):
    if f < 20:
        return 0.0
    elif 20 <= f < 30:
        return 0.6 * (f - 20) / 10.0
    elif 30 <= f < 60:
        return 0.6 + 0.4 * (np.log(f / 30.0) / np.log(60.0 / 30.0))
    else:
        return 1.0

def calc_norm_power(p):
    P_IDEAL = 6.0
    P_LIMIT = 10.0
    if p <= P_IDEAL:
        return 1.0
    elif P_IDEAL < p <= P_LIMIT:
        return 0.6 + 0.4 * (((P_LIMIT - p) / (P_LIMIT - P_IDEAL)) ** 2)
    else:
        return 0.0

# 6. 应用计算公式
print("Calculating WNS scores...")
df_calc['Norm_mAP'] = df_calc['mAP50'].apply(calc_norm_map)
df_calc['Norm_FPS'] = df_calc['FPS_Mean'].apply(calc_norm_fps)
df_calc['Norm_Power'] = df_calc['Power_Mean'].apply(calc_norm_power)

df_calc['Final_WNS_Score'] = (
    0.8 * df_calc['Norm_mAP'] + 
    0.1 * df_calc['Norm_FPS'] + 
    0.1 * df_calc['Norm_Power']
)

# 7. 整理并格式化输出列
# 将名字拆分回去，方便你对标 LaTeX 表格
df_calc['Architecture'] = df_calc['Model_Name'].apply(lambda x: x.split('-')[0])
df_calc['Format'] = df_calc['Model_Name'].apply(lambda x: x.split('-')[1])

output_cols = [
    'Architecture', 'Format', 
    'mAP50', 
    'FPS_Mean', 'FPS_Std', 
    'Power_Mean', 'Power_Std', 
    'Norm_mAP', 'Norm_FPS', 'Norm_Power', 
    'Final_WNS_Score'
]
df_final = df_calc[output_cols].copy()

# 按总分降序排列，方便你直接看到排名前十
df_final = df_final.sort_values(by='Final_WNS_Score', ascending=False)

# 8. 导出到 Excel
print(f"Exporting results to {OUTPUT_EXCEL}...")
df_final.to_excel(OUTPUT_EXCEL, index=False, float_format="%.4f")
print("Done! Please check the Excel file to verify the calculations.")