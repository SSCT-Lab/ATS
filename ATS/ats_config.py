is_radius_th = False
radius_th = 0.00001  # 每个点的距离小于此值认为没有覆盖

boundary = 0  # 下界 此下界<x< 上界的 进行覆盖分析  x<此下界的待定
up_boundary = 0.99  # 上界 >此上界的认为该测试用例没价值

is_log = False  # 距离函数是否使用log
linear_ratio = 0.005  # 每个点的覆盖半径 ratio *d 线性
log_ratio = 0.01

round_num = 5  # 对端点进行浮点数约束,最多为5位
# 选择算法变体 not use
# select_pnum = 100  # not use
# step_th = 0.005  # not useF
