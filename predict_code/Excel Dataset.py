import pandas as pd
import numpy as np


def classify_angina(r):
    if r['CDQ001'] != 1 or r['CDQ002'] != 1 or r['CDQ004'] != 1 or r['CDQ005'] != 1 or r['CDQ006'] != 1:
        # 无病
        return 0

    if r['CDQ003'] != 1:
        # 检查1级心绞痛的条件
        if (r['CDQ009D'] == 4 or r['CDQ009E'] == 5) or (r['CDQ009F'] == 6 and r['CDQ009G'] == 7):
            return 1
    else:
        # 检查2级心绞痛的条件
        if (r['CDQ009D'] == 4 or r['CDQ009E'] == 5) or (r['CDQ009F'] == 6 and r['CDQ009G'] == 7):
            return 2
    return 0

def classify_Disease(r):
    num = 0
    Lst = ['MCQ160B','MCQ160C','MCQ160D','MCQ160E','MCQ160F']
    Disease_Dict = {
        'MCQ160B':1,
        'MCQ160C':2,
        'MCQ160D':3,
        'MCQ160E':4,
        'MCQ160F':5
    }
    for i in Lst:
        if r[i] == 1:
            num+=1
            Disease = Disease_Dict[i]
        if num > 1:
            Disease = 6
        elif num == 0:
            Disease = 0
    return Disease
#数据读取
sheet_name = "中度-重度-增殖 NPR"
df = pd.read_excel('NHANES Total v1.1(No Dustry).xlsx', sheet_name=sheet_name, na_values=[' '])

#缺失数据补充
for column in df.columns:
    print(df[column].isnull().any())
    if df[column].isnull().any():
        if column == 'OPDDVCDR' or column == 'OPDSVCDR':
            mean_value = round(df[column].mean(), 3)
        else:
            mean_value = int(df[column].mean())
        df[column].fillna(mean_value, inplace=True)
        print(f"列 {column} 的缺失值已用均值 {mean_value} 填充")

output_file = f"FinalInput_{sheet_name}.xlsx"
df.to_excel(output_file, index=False)
print(f"处理完成")



# print(data.index)
# print(data[0:500])
# print("-------------------------")
#





