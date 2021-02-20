import math
import numpy as np
import pandas as pd

# 产生数据
# motor1_current = ['normal', 'increase', 'decrease']
# motor2_current = ['normal', 'increase', 'decrease']
# motor3_current = ['normal', 'increase', 'decrease']
# motor4_current = ['normal', 'increase', 'decrease']
# motor5_current = ['normal', 'increase', 'decrease']
# motor6_current = ['normal', 'increase', 'decrease']
# total_coal_quality = ['normal', 'increase', 'decrease']
# instantaneous_coal_quality = ['normal', 'increase', 'decrease']

# 设定输入特征的取值范围
motor1_current = [1, 2, 3]
motor2_current = [1, 2, 3]
motor3_current = [1, 2, 3]
motor4_current = [1, 2, 3]
motor5_current = [1, 2, 3]
motor6_current = [1, 2, 3]
total_coal_quality = [1, 2, 3]
instantaneous_coal_quality = [1, 2, 3]


motor1_error = [0, 1]
motor2_error = [0, 1]
motor3_error = [0, 1]
motor4_error = [0, 1]
motor5_error = [0, 1]
motor6_error = [0, 1]

# 得到前8个特征所有的可能组合
input_data1 = []
for a in motor1_current:
    for b in motor2_current:
        for c in  motor3_current:
            for d in motor4_current:
                for e in motor5_current:
                    for f in motor6_current:
                        for g in total_coal_quality:
                            for h in instantaneous_coal_quality:
                                input_data1.append((a, b, c, d, e, f, g, h))

# 得到后6个特征所有的可能组合
input_data2 = []
for i in motor1_error:
    for j in motor2_error:
        for k in motor3_error:
            for l in motor4_error:
                for m in motor5_error:
                    for n in motor6_error:
                        input_data2.append((i, j, k, l, m, n))
print(input_data1)
print(input_data2)
row1 = (math.pow(3, 8))
row2 = (math.pow(2, 6))
input_data_part1 = np.zeros((8, int(row1)))
input_data_part1 = np.array(input_data1)
input_data_part2 = np.zeros((8, int(row2)))
input_data_part2 = np.array(input_data2)
print(input_data_part1)
print(input_data_part2)

dataframe1 = pd.DataFrame({'motor1_current': input_data_part1[:, 0],
                          'motor2_current': input_data_part1[:, 1],
                          'motor3_current': input_data_part1[:, 2],
                          'motor4_current': input_data_part1[:, 3],
                          'motor5_current': input_data_part1[:, 4],
                          'motor6_current': input_data_part1[:, 5],
                          'total_coal_quality': input_data_part1[:, 6],
                          'instantaneous_coal_quality': input_data_part1[:, 7]})
dataframe2 = pd.DataFrame({'motor1_error': input_data_part2[:, 0],
                          'motor2_error': input_data_part2[:, 1],
                          'motor3_error': input_data_part2[:, 2],
                          'motor4_error': input_data_part2[:, 3],
                          'motor5_error': input_data_part2[:, 4],
                          'motor6_error': input_data_part2[:, 5]})

# 保存到csv文档
dataframe1.to_csv("input_part1.csv")
dataframe2.to_csv("input_part2.csv")