import sys
import numpy
import logging
import time

logging.basicConfig(level=logging.INFO)  # 打印日志


# region 读取并处理数据
def finish():
    """
    写入OK结束并发送缓存。
    :return: none
    """
    sys.stdout.write('OK\n')
    sys.stdout.flush()  # 有这条才会把内存中的数据传给判题器


def get_init_data(raw_data):
    """
    获取初始化数据并处理。
    :param raw_data: 接收的原始数据，调用judge_machine()或local_test()
    :return: none
    """
    global bot_list, table_list, map_num
    global table_list
    for rows in range(len(raw_data)):  # 迭代每个字符串和每个字符，检查是否为工作台或机器人,返回具体坐标
        for column in range(len(raw_data[rows])):
            char = raw_data[rows][column]
            if char == 'A':
                x = (column + 1) / 2 - 0.25  # 1.+1是因为range从0开始  2.地图实际坐标与遍历顺序颠倒
                y = (100 - rows) / 2 - 0.25  # 3.最上面是50所以y要处理一下  (101-y)/2 -0.25
                # 机器人数据字典
                temp_bot_dict = {'table_id': -1,  # 可交互工作台ID
                                 'object': 0,  # 持有物品类别
                                 'time_factor': 0,  # 时间价值系数
                                 'collision_factor': 0,  # 碰撞价值系数
                                 'angular_velocity': 0,  # 角速度
                                 'linear_velocity': (0, 0),  # 线速度
                                 'direction': 0,  # 朝向
                                 'position': (x, y),  # 坐标
                                 'work_status': 0,  # 工作状态，0：空闲，1：正在前往第一个目标工作台，2：正在前往第二个目标工作台
                                 # 'motion_status': 0,  # 期待运动状态，0：静止，1：旋转，2：前进（基于PID算法使用，已废弃）
                                 'target_table_id': (),  # 目标工作台ID
                                 }
                bot_list.append(temp_bot_dict)
            elif char in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                x = (column + 1) / 2 - 0.25
                y = (100 - rows) / 2 - 0.25
                # 工作台数据字典
                temp_table_dict = {'type': int(char),  # 类别
                                   'position': (x, y),  # 坐标
                                   'remaining_time': -1,  # 剩余生产时间
                                   'pre_remaining_time': -1,  # 上一帧的剩余生产时间
                                   'material': 0,  # 原材料格状态（原始数据）
                                   'pre_material': 0,  # 上一帧的原材料格状态（原始数据）
                                   'production': 0,  # 产品格状态
                                   'pre_production': 0,  # 上一帧的产品格状态
                                   'missing_material': (),  # 缺少的原材料物品类别
                                   'projected_missing_material': [],  # 考虑计划的缺少的原材料物品类别
                                   'projected_production': 0,  # 考虑计划的产品格状态
                                   }
                table_list.append(temp_table_dict)
    get_map_num()
    ban_table()
    generate_index()  # 生成工作台类别索引字典
    generate_type()  # 生成工作台类别列表
    finish()  # 结束


def judge_machine():
    """
    get_init_data()的参数，接入判题器时使用。
    :return: 判题器发送的初始化原始数据
    """
    temp_raw_data = []  # 存放判题器数据
    while True:
        line = sys.stdin.readline()  # 按行传的,结尾有一个换行符\n
        if line != 'OK\n':
            temp_raw_data.append(line)
        else:
            break
    return temp_raw_data


# def local_test():
#     """
#     get_init_data()的参数，使用本地数据集调试时使用
#     :return: 本地数据集初始化原始数据
#     """
#     with open('a.txt', 'r') as file:
#         raw_data = file.readlines()
#     return raw_data


def generate_type():
    """
    生成工作台种类列表。
    :return: none
    """
    global table_type
    global table_list
    for type in table_list:
        table_type.append(int(type['type']))
    table_type = list(set(table_type))
    if 0 in table_type:
        table_type.remove(0)


def generate_index():
    """
    生成工作台类别索引字典。
    :return: none
    """
    global bot_list
    global table_list
    global table_index
    for num in range(0, 10):  # 生成0-9的列表
        table_index[f'type_{num}'] = []
    for index, table in enumerate(table_list):
        type = table['type']
        table_index[f'type_{type}'].append(index)


def update_data():
    """
    更新并处理帧数据。
    :return: none
    """
    global frame_id
    global money
    global bot_list
    global table_list
    global table_num
    temp_list = []  # 临时存放帧数据,每一帧的每一行都是一个列表
    # [['第一行第一个数据','第一行第二个数据'],['第二行第一个数据','第二行第二个数据', -----]----]
    while True:
        temp_line = input()  # input()获取到的数据没有带 \n 可以直接切开,如果用sys获取,最后一个数据会有一个\n,不好切
        if temp_line != 'OK':
            temp_list.append(temp_line.split(' '))  # 以空格分割出每个元素
            # logging.info(temp_line)  # 查看判题器的原始数据
        else:
            break
    frame_id = int(temp_list[0][0])  # 帧id,发送用
    money = int(temp_list[0][1])  # 当前金钱数
    table_num = int(temp_list[1][0])
    bot_update_data = temp_list[-4:]  # 后四个是机器人的四个列表
    table_update_data = temp_list[2:-4]  # 工作台部分
    for i in range(4):  # 从0开始,机器人只有四个
        old_bot = bot_list[i]  # 拿出旧的
        new_bot = bot_update_data[i]  # 对应位置的新的
        old_bot['table_id'] = int(new_bot[0])
        old_bot['object'] = int(new_bot[1])
        old_bot['time_factor'] = float(new_bot[2])
        old_bot['collision_factor'] = float(new_bot[3])
        old_bot['angular_velocity'] = float(new_bot[4])
        old_bot['linear_velocity'] = (float(new_bot[5]), float(new_bot[6]))
        old_bot['direction'] = (float(new_bot[7]))
        old_bot['position'] = (float(new_bot[8]), float(new_bot[9]))
    for i in range(table_num):  # 工作台的数量不固定,需要用长度确定循环次数
        old_table = table_list[i]
        new_table = table_update_data[i]
        old_table['pre_remaining_time'] = old_table['remaining_time']
        old_table['remaining_time'] = int(new_table[3])
        old_table['pre_material'] = old_table['material']
        old_table['material'] = int(new_table[4])
        old_table['pre_production'] = old_table['production']
        old_table['production'] = int(new_table[5])
    # 更新完后计算每种类型缺失的材料
    missing_material_dict = {
        (4, 0): (1, 2),
        (4, 2): (2),
        (4, 4): (1),
        (4, 6): (),
        (5, 0): (1, 3),
        (5, 2): (3),
        (5, 8): (1),
        (5, 10): (),
        (6, 0): (2, 3),
        (6, 4): (3),
        (6, 8): (2),
        (6, 12): (),
        (7, 0): (4, 5, 6),
        (7, 16): (5, 6),
        (7, 32): (4, 6),
        (7, 64): (4, 5),
        (7, 48): (6),
        (7, 80): (5),
        (7, 112): ()
    }
    for i, table in enumerate(table_list):
        type = table['type']
        own = table['material']
        table_list[i]['missing_material'] = missing_material_dict.get((type, own), ())
    # 生成计划产品格任务
    for i, table in enumerate(table_list):
        if table['pre_remaining_time'] == 0:
            if table['type'] not in [8, 9]:
                if table['remaining_time'] != 0:
                    table_list[i]['projected_production'] = 1
        # 仍有一种剩余1帧被取走的情况,后续可优化
        else:
            if table['type'] not in [8, 9]:
                if table['production'] == 1 and table['pre_production'] == 0:
                    table_list[i]['projected_production'] = 1
    # 生成计划材料格任务
    for i, table in enumerate(table_list):
        init = bool(frame_id == 1)  # 用于第一帧初始化任务列表
        change = bool(table['pre_material'] > 0 and (table['material'] == 0))  # 用于后续帧判断
        if table['type'] == 4:
            if change or init:
                table_list[i]['projected_missing_material'] = [1, 2]
        elif table['type'] == 5:
            if change or init:
                table_list[i]['projected_missing_material'] = [1, 3]
        elif table['type'] == 6:
            if change or init:
                table_list[i]['projected_missing_material'] = [2, 3]
        elif table['type'] == 7:
            if change or init:
                table_list[i]['projected_missing_material'] = [4, 5, 6]


# endregion


# region 发送指令
def send_data():
    """
    解析指令字典并发送指令。
    :return: none
    """
    global command
    global frame_id
    sys.stdout.write('%d\n' % frame_id)
    for robot_id in range(4):
        sys.stdout.write('forward %d %d\n' % (robot_id, command['forward'][robot_id]))
        sys.stdout.write('rotate %d %d\n' % (robot_id, command['rotate'][robot_id]))
    if command['sell']:
        for bot_id in command['sell']:
            sys.stdout.write('sell %d\n' % bot_id)
        command['sell'] = []
    if command['buy']:
        for bot_id in command['buy']:
            sys.stdout.write('buy %d\n' % bot_id)
        command['buy'] = []
    if command['destroy']:
        for bot_id in command['destroy']:
            sys.stdout.write('destroy %d\n' % bot_id)
        command['destroy'] = []
    finish()


# endregion


# region PID控制算法，已废弃
# class PID(object):
#
#     def __init__(self):
#         self.kp = 0
#         self.ki = 0
#         self.kd = 0
#         self.current_val = 0
#         self.expected_val = 0
#         self.sum_error = 0
#         self.last_error = 0
#
#     def set(self, kp, ki, kd, initial_val, expected_val):
#         """
#         设置PID算法参数以及被控量的初始值与目标值。
#         :param kp: 比例项系数
#         :param ki: 积分项系数
#         :param kd: 微分项系数
#         :param initial_val: 被控量初始值
#         :param expected_val: 被控量目标值
#         :return: none
#         """
#         self.kp = kp
#         self.ki = ki
#         self.kd = kd
#         self.current_val = initial_val
#         self.expected_val = expected_val
#         self.sum_error = 0
#         self.last_error = 0
#
#     def update(self, current_val):
#         """
#         更新当前值。
#         :param current_val: 当前值
#         :return: none
#         """
#         self.current_val = current_val
#
#     def calculate(self):
#         """
#         计算控制量。
#         :return: 控制量
#         """
#         error = self.expected_val - self.current_val
#         self.sum_error += error
#         output = self.kp * error + self.ki * self.sum_error + self.kd * (error - self.last_error)
#         self.last_error = error
#         return output
#
#
# # PID控制参数
# linear_velocity_PID_param = {
#     'no_load_kp': 0,
#     'no_load_ki': 0,
#     'no_load_kd': 0,
#     'on_load_kp': 0,
#     'on_load_ki': 0,
#     'on_load_kd': 0,
# }
# angular_velocity_PID_param = {
#     'no_load_kp': 5,
#     'no_load_ki': 0.2,
#     'no_load_kd': 5,
#     'on_load_kp': 10,
#     'on_load_ki': 0.4,
#     'on_load_kd': 10,
# }
#
# # PID控制对象
# bot_linear_velocity_PID = [PID(), PID(), PID(), PID()]
# bot_angular_velocity_PID = [PID(), PID(), PID(), PID()]


# endregion


# region 基于PID算法的运动控制与附属功能函数，已废弃
# def rotate(bot_id: int, table_id: int):
#     """
#     使用PID算法使机器人原地旋转到面朝指定工作台的方向，不含停止旋转条件控制。
#     :param bot_id: 机器人ID
#     :param table_id: 目标工作台ID
#     :return: none
#     """
#     global bot_list, table_list, command
#     bot = bot_list[bot_id]
#     # 期望运动状态处于静止状态，则设定转动目标
#     if bot['motion_status'] == 0:
#         expected_angle = calculate_angle(bot_id, table_id)  # 计算目标方向角
#         # 目标方向和当前方向之差大于pi时需要转换
#         if numpy.abs(expected_angle - bot['direction']) > numpy.pi:
#             # 当前朝向先做转换变到与目标方向同号
#             # 当前朝向方向角为0不需要转换
#             # 转换结果只在临时变量中用于输入PID，并不修改数据字典中的值
#             if bot['direction'] < 0:
#                 bot['direction'] += 2 * numpy.pi
#             elif bot['direction'] > 0:
#                 bot['direction'] -= 2 * numpy.pi
#         # 设置PID算法参数与朝向初始值与目标值
#         if bot['object'] == 0:
#             bot_angular_velocity_PID[bot_id].set(angular_velocity_PID_param['no_load_kp'],
#                                                  angular_velocity_PID_param['no_load_ki'],
#                                                  angular_velocity_PID_param['no_load_kd'], bot['direction'],
#                                                  expected_angle)
#         else:
#             bot_angular_velocity_PID[bot_id].set(angular_velocity_PID_param['on_load_kp'],
#                                                  angular_velocity_PID_param['on_load_ki'],
#                                                  angular_velocity_PID_param['on_load_kd'], bot['direction'],
#                                                  expected_angle)
#         # 计算控制量并设置为角速度
#         command['rotate'][bot_id] = bot_angular_velocity_PID[bot_id].calculate()
#         # 期望运动状态修改为旋转态
#         bot_list[bot_id]['motion_status'] = 1
#     # 期望运动状态处于旋转状态，则持续进行PID算法控制
#     elif bot['motion_status'] == 1:
#         # 目标方向和当前方向之差大于pi时需要转换
#         if numpy.abs(bot_angular_velocity_PID[bot_id].expected_val - bot['direction']) > numpy.pi:
#             # 当前朝向先做转换变到与目标方向同号
#             # 当前朝向方向角为0不需要转换
#             # 转换结果只在临时变量中用于输入PID，并不修改数据字典中的值
#             if bot['direction'] < 0:
#                 bot['direction'] += 2 * numpy.pi
#             elif bot['direction'] > 0:
#                 bot['direction'] -= 2 * numpy.pi
#         # 更新当前值到PID
#         bot_angular_velocity_PID[bot_id].update(bot['direction'])
#         # 计算控制量并设置为角速度
#         command['rotate'][bot_id] = bot_angular_velocity_PID[bot_id].calculate()
#         logging.info(bot_angular_velocity_PID[bot_id].expected_val)
#         logging.info(bot_list[bot_id]['direction'])
#
#
# def start_forward(bot_id: int):
#     """
#     开始前进。
#     :param bot_id: 机器人ID
#     :return: none
#     """
#     global bot_list, table_list, command
#     bot = bot_list[bot_id]
#     # 期望运动状态处于静止状态，则开始以最大速度前进
#     if bot['motion_status'] == 0:
#         command['forward'][bot_id] = max_forward_velocity
#         # 期望运动状态修改为前进
#         bot_list[bot_id]['motion_status'] = 1
#
#
# def move(bot_id: int, table_id: int):
#     """
#     使用PID算法使机器人原地旋转到目标工作台的方向然后前进，不含刹车。
#     :param bot_id: 机器人ID
#     :param table_id: 工作台ID
#     :return: none
#     """
#     global bot_list, table_list
#     angle_threshold = 0.01  # 角度误差阈值
#     angular_velocity_threshold = 0.05  # 停止角速度阈值
#     bot = bot_list[bot_id]
#     # 角度误差小于阈值且角速度低于停止角速度阈值时停止旋转，开始前进
#     if numpy.abs(bot['direction'] - calculate_angle(bot_id, table_id)) < angle_threshold and bot[
#         'angular_velocity'] < angular_velocity_threshold:
#         if bot['motion_status'] == 1:
#             command['rotate'][bot_id] = 0
#             bot['motion_status'] = 0
#             start_forward(bot_id)
#     # 继续旋转
#     if bot['motion_status'] != 2:
#         rotate(bot_id, table_id)
#
#
# def buy():
#     global bot_list, command
#     for bot_id, bot in enumerate(bot_list):
#         # 有任务才进来，没任务的时候目标工作台ID元组为空会报错
#         if bot['work_status'] != 0:
#             # 所处工作台ID和第一个目标工作台（购买）ID相同且手上没有物品
#             if bot['table_id'] == bot['target_table_id'][0] != -1 and bot['object'] == 0:
#                 # 写入购买指令
#                 command['buy'].append(bot_id)
#                 # 刹车
#                 command['forward'][bot_id] = 0
#                 # 期望运动状态修改为静止
#                 bot_list[bot_id]['motion_status'] = 0
#
#
# def sell():
#     global bot_list, command
#     for bot_id, bot in enumerate(bot_list):
#         # 有任务才进来，没任务的时候目标工作台ID元组为空会报错
#         if bot['work_status'] != 0:
#             # 所处工作台ID和第二个目标工作台（出售）ID相同且手上有物品
#             if bot['table_id'] == bot['target_table_id'][1] != -1 and bot['object'] != 0:
#                 # 写入出售指令
#                 command['sell'].append(bot_id)
#                 # 刹车
#                 command['forward'][bot_id] = 0
#                 # 期望运动状态修改为静止
#                 bot_list[bot_id]['motion_status'] = 0


# endregion


# region 几何运算
def calculate_angle(bot_id: int, table_id: int):
    """
    计算机器人与工作台连线的方向角。
    :param bot_id: 机器人ID
    :param table_id: 工作台ID
    :return: 机器人与工作台连线的方向角
    """
    global bot_list, table_list
    bot_position = bot_list[bot_id]['position']  # 机器人坐标
    table_position = table_list[table_id]['position']  # 目的工作台坐标
    relative_position = (table_position[0] - bot_position[0], table_position[1] - bot_position[1])  # 工作台相对于机器人的坐标
    angle = numpy.arctan2(relative_position[1], relative_position[0])  # 工作台相对于机器人的方向角，(-pi,pi]
    return angle


def calculate_angle_between_bot(bot_id_1: int, bot_id_2: int):
    """
    计算机器人2相对于机器人1的方向角。
    :param bot_id_1: 机器人1ID
    :param bot_id_2: 机器人2ID
    :return: 机器人2相对于机器人1的方向角
    """
    global bot_list
    bot_1_position = bot_list[bot_id_1]['position']  # 机器人1坐标
    bot_2_position = bot_list[bot_id_2]['position']  # 机器人2坐标
    relative_position = (bot_2_position[0] - bot_1_position[0], bot_2_position[1] - bot_1_position[1])  # 机器人2相对于机器人1的坐标
    angle = numpy.arctan2(relative_position[1], relative_position[0])  # 机器人2相对于机器人1的方向角，(-pi,pi]
    return angle


def calculate_distance(bot_id: int, table_id: int):
    """
    计算机器人与工作台的距离。
    :param bot_id: 机器人ID
    :param table_id: 工作台ID
    :return: 机器人与工作台的距离
    """
    global bot_list, table_list
    distance = numpy.sqrt((bot_list[bot_id]['position'][0] - table_list[table_id]['position'][0]) ** 2 + (
            bot_list[bot_id]['position'][1] - table_list[table_id]['position'][1]) ** 2)
    return distance


def calculate_distance_between_table(table_id_1: int, table_id_2: int):
    """
    计算工作台之间的距离。
    :param table_id_1: 工作台1ID
    :param table_id_2: 工作台2ID
    :return: 两工作台之间的距离
    """
    global table_list
    distance = numpy.sqrt((table_list[table_id_1]['position'][0] - table_list[table_id_2]['position'][0]) ** 2 + (
            table_list[table_id_1]['position'][1] - table_list[table_id_2]['position'][1]) ** 2)
    return distance


def calculate_distance_between_bot(bot_id_1: int, bot_id_2: int):
    """
    计算机器人之间的距离。
    :param bot_id_1: bot_id_1: 机器人1ID
    :param bot_id_2: bot_id_2: 机器人2ID
    :return: 两机器人之间的距离
    """
    global bot_list
    distance = numpy.sqrt((bot_list[bot_id_1]['position'][0] - bot_list[bot_id_2]['position'][0]) ** 2 + (
            bot_list[bot_id_1]['position'][1] - bot_list[bot_id_2]['position'][1]) ** 2)
    return distance


# endregion


# region 运动控制
def move(bot_id: int, table_id: int):
    """
    控制机器人移动到目标工作台。
    :param bot_id: 机器人ID
    :param table_id: 目标工作台ID
    :return: none
    """
    global bot_list
    angular_velocity_kp = 100  # 纠正角度偏差的比例系数
    deceleration_distance_threshold = 1  # 减速距离阈值
    deceleration_kp = 0.5  # 到达减速距离阈值后速度比例系数
    start_move_angle_threshold = numpy.pi / 3  # 开始移动偏差角阈值
    bot = bot_list[bot_id]
    # 方向调整控制
    expected_angle = calculate_angle(bot_id, table_id)  # 计算目标方向角
    # 目标方向和当前方向之差大于pi时需要转换
    if numpy.abs(expected_angle - bot['direction']) > numpy.pi:
        # 当前朝向先做转换变到与目标方向同号
        # 当前朝向方向角为0不需要转换
        # 转换结果只在临时变量中用于计算偏差角，并不修改数据字典中的值
        if bot['direction'] < 0:
            bot['direction'] += 2 * numpy.pi
        elif bot['direction'] > 0:
            bot['direction'] -= 2 * numpy.pi
    # 控制旋转速度，偏差角度越大角速度越大
    # if expected_angle - bot['direction'] < 0:
    #     command['rotate'][bot_id] = -min_rotate_velocity + angular_velocity_kp * (expected_angle - bot['direction'])
    # elif expected_angle - bot['direction'] > 0:
    #     command['rotate'][bot_id] = min_rotate_velocity + angular_velocity_kp * (expected_angle - bot['direction'])
    # 理论上来说是上面这样的，但是不知道为什么下面这个有疏漏的控制式会更快
    command['rotate'][bot_id] = min_rotate_velocity + angular_velocity_kp * (expected_angle - bot['direction'])
    # 到达偏差角阈值以最大速度前进
    if numpy.abs(expected_angle - bot['direction']) < start_move_angle_threshold:
        command['forward'][bot_id] = max_forward_velocity
    # 否则以最小速度前进
    else:
        command['forward'][bot_id] = 1
    # 前进速度控制
    distance = calculate_distance(bot_id, table_id)  # 计算与目的地的距离
    # 与目的地距离近的时候减速
    if distance < deceleration_distance_threshold:
        command['forward'][bot_id] = max_forward_velocity * deceleration_kp


# endregion


# region 买卖控制
def buy():
    """
    检查是否需要进行购买操作，需要的话则进行购买。
    :return: none
    """
    global bot_list, command
    for bot_id, bot in enumerate(bot_list):
        # 正在前往第一个目标工作台时才判断
        if bot['work_status'] == 1:
            # 所处工作台ID和第一个目标工作台（购买）ID相同且手上没有物品
            if bot['table_id'] == bot['target_table_id'][0] != -1 and bot['object'] == 0:
                # 写入购买指令
                command['buy'].append(bot_id)


def buy_check():
    """
    检查上一帧是否购买成功，若购买成功则修改工作状态为前往第二个目标工作台。
    :return: none
    """
    global bot_list, command
    for bot_id, bot in enumerate(bot_list):
        # 正在前往第一个目标工作台时才判断
        if bot['work_status'] == 1:
            # 持有物品是要买的物品则为购买成功
            if bot['object'] == table_list[bot['target_table_id'][0]]['type']:
                # 刹车
                command['forward'][bot_id] = 0
                # 角速度设置为0
                command['rotate'][bot_id] = 0
                # 修改工作状态到前往第二个目标工作台
                bot_list[bot_id]['work_status'] = 2


def sell():
    """
    检查是否需要进行出售操作，需要的话则进行出售。
    :return: none
    """
    global bot_list, command
    for bot_id, bot in enumerate(bot_list):
        # 正在前往第二个目标工作台时才判断
        if bot['work_status'] == 2:
            # 所处工作台ID和第二个目标工作台（出售）ID相同且手上有物品
            if bot['table_id'] == bot['target_table_id'][1] != -1 and bot['object'] != 0:
                # 写入出售指令
                command['sell'].append(bot_id)


def sell_check():
    """
    检查上一帧是否出售成功，若出售成功则修改工作状态为空闲。
    :return: none
    """
    global bot_list, command
    for bot_id, bot in enumerate(bot_list):
        # 正在前往第二个目标工作台时才判断
        if bot['work_status'] == 2:
            # 已经没有持有物品则为出售成功
            if bot['object'] == 0:
                # 刹车
                command['forward'][bot_id] = 0
                # 角速度设置为0
                command['rotate'][bot_id] = 0
                # 修改工作状态到空闲
                bot_list[bot_id]['work_status'] = 0


# endregion


# region 机器人总控
def control():
    """
    机器人运动控制总函数，使机器人按照任务进行移动、购买、出售。
    :return: none
    """
    global bot_list
    # 运动控制
    for bot_id, bot in enumerate(bot_list):
        # 有任务才进来，没任务的时候目标工作台ID元组为空会报错
        if bot['work_status'] != 0:
            move(bot_id, bot['target_table_id'][bot['work_status'] - 1])
    buy()  # 检查是否可以购买并购买
    sell()  # 检查是否可以出售并出售，更新工作状态到空闲
    buy_check()  # 检查是否购买成功（在购买的下一帧起效），并更新工作状态到前往第二个目标工作台
    sell_check()  # 检查是否出售成功（在出售的下一帧起效），并更新工作状态到空闲


# endregion


# region 防碰撞
# 基于距离的超级简陋防碰撞算法，垃圾一个，已废弃
# def collision_prediction():
#     """
#     碰撞预测，即将可能发生碰撞的两个机器人将有一个会减速避让。暂不考虑撞墙。
#     :return: none
#     """
#     global bot_list, command
#     threshold_value = 1.5  # 预计碰撞的距离阈值
#     avoidance_velocity = max_forward_velocity / 2
#     # 取机器人两两组合
#     for bot_id_1 in range(bot_num - 1):
#         for bot_id_2 in range(bot_id_1 + 1, bot_num):
#             bot_1 = bot_list[bot_id_1]
#             bot_2 = bot_list[bot_id_2]
#             # 两个机器人都没有持有物品不需要防碰撞
#             if bot_1['object'] == bot_2['object'] == 0:
#                 continue
#             # 计算两两间距离
#             distance = numpy.sqrt(
#                 (bot_1['position'][0] - bot_2['position'][0]) ** 2 + (bot_1['position'][1] - bot_2['position'][1]) ** 2)
#             # 两机器人之间距离小于阈值
#             if distance < threshold_value:
#                 # 手上物品价值小的停止运动进行避让，物品类型优先，物品类型相同则算价值系数
#                 if bot_1['object'] < bot_2['object']:
#                     # 修改指令字典该机器人速度为避让速度
#                     command['forward'][bot_id_1] = avoidance_velocity
#                 elif bot_1['object'] > bot_2['object']:
#                     # 修改指令字典该机器人速度为避让速度
#                     command['forward'][bot_id_2] = avoidance_velocity
#                 else:
#                     if bot_1['time_factor'] * bot_1['collision_factor'] < bot_2['time_factor'] * bot_2[
#                         'collision_factor']:
#                         # 修改指令字典该机器人速度为避让速度
#                         command['forward'][bot_id_1] = avoidance_velocity
#                     else:
#                         # 修改指令字典该机器人速度为避让速度
#                         command['forward'][bot_id_2] = avoidance_velocity


# 更改为可传参版
# def collision_prediction():
#     """
#     简易的单目标防碰撞。
#     :return: none
#     """
#     global bot_list, command
#     angular_velocity_kp = 100  # 纠正角度偏差的比例系数
#     min_distance_kp = 1.4  # 距离大于圆心距乘该系数才触发避障
#     max_distance_kp = 100  # 距离小于圆心距乘该系数才触发避障
#     collision_area_half_angle_kp = 1.2  # 增大碰撞区的系数
#     # 单目标防碰撞，遍历每个主动避让机器人
#     for bot_id_1, bot_1 in enumerate(bot_list):
#         distance_list = []  # 机器人距离列表
#         # 遍历其他三个机器人并计算与主动避让机器人的距离（但是为了保证下标一致也算上自身）
#         for bot_id_2 in range(bot_num):
#             # 计算距离
#             distance = calculate_distance_between_bot(bot_id_1, bot_id_2)
#             distance_list.append(distance)
#         # 得出由近到远的索引数组
#         distance_sort_index = numpy.argsort(distance_list)
#         # 按由近到远的顺序遍历其他三个机器人
#         for bot_id_2 in distance_sort_index:
#             # 不算自己
#             if bot_id_2 != bot_id_1:
#                 bot_2 = bot_list[bot_id_2]
#                 # 获取机器人半径
#                 if bot_1['object'] == 0:
#                     bot_1_radius = no_load_radius
#                 else:
#                     bot_1_radius = on_load_radius
#                 if bot_2['object'] == 0:
#                     bot_2_radius = no_load_radius
#                 else:
#                     bot_2_radius = on_load_radius
#                 # 触发避障的距离
#                 if (bot_1_radius + bot_2_radius) * min_distance_kp < distance_list[bot_id_2] < (
#                         bot_1_radius + bot_2_radius) * max_distance_kp:
#                     # 计算碰撞区
#                     collision_area_half_angle = numpy.arcsin(
#                         (bot_1_radius + bot_2_radius) / distance_list[bot_id_2])  # 碰撞区半角
#                     # 适当增加碰撞区半角，但不超出最大值
#                     if collision_area_half_angle * collision_area_half_angle_kp > numpy.pi / 2:
#                         collision_area_half_angle = numpy.pi / 2
#                     else:
#                         collision_area_half_angle *= collision_area_half_angle_kp
#                     collision_area_center_angle = calculate_angle_between_bot(bot_id_1, bot_id_2)  # 碰撞区中心方向角
#                     collision_area_left_boundary_angle = collision_area_center_angle + collision_area_half_angle  # 碰撞区左边界方向角
#                     collision_area_right_boundary_angle = collision_area_center_angle - collision_area_half_angle  # 碰撞区右边界角
#                     # 计算机器人1相对于机器人2的速度
#                     bot_1_relative_velocity_x = bot_1['linear_velocity'][0] - bot_2['linear_velocity'][0]  # x方向相对速度
#                     bot_1_relative_velocity_y = bot_1['linear_velocity'][1] - bot_2['linear_velocity'][1]  # y方向相对速度
#                     bot_1_relative_velocity_angle = numpy.arctan2(bot_1_relative_velocity_y,
#                                                                   bot_1_relative_velocity_x)  # 相对速度方向
#                     # 如果相对速度方向和碰撞区中心方向角之差大于pi（例如碰撞区是负角度，但相对速度方向在其+2pi后的正角度区间内）
#                     if numpy.abs(bot_1_relative_velocity_angle - collision_area_center_angle) > numpy.pi:
#                         # 变换到同号区间
#                         if bot_1_relative_velocity_angle < 0:
#                             bot_1_relative_velocity_angle += 2 * numpy.pi
#                         elif bot_1_relative_velocity_angle > 0:
#                             bot_1_relative_velocity_angle -= 2 * numpy.pi
#                     # 如果相对速度方向在碰撞区内则会发生碰撞（恰巧在边界会相切，所以也避免），需要避让
#                     if collision_area_right_boundary_angle < bot_1_relative_velocity_angle < collision_area_left_boundary_angle:
#                         # 检测离哪边边界近就往哪边转，直接和中心方向角比较即可
#                         # 离右边界近
#                         if bot_1_relative_velocity_angle < collision_area_center_angle:
#                             command['rotate'][bot_id_1] = min_rotate_velocity + angular_velocity_kp * (
#                                     collision_area_right_boundary_angle - bot_1_relative_velocity_angle)
#                         # 离左边界近
#                         elif bot_1_relative_velocity_angle > collision_area_center_angle:
#                             command['rotate'][bot_id_1] = min_rotate_velocity + angular_velocity_kp * (
#                                     collision_area_left_boundary_angle - bot_1_relative_velocity_angle)
#                         # 恰好位于中间
#                         else:
#                             # 查看目前的角速度方向，往当前已有的角速度方向转，角速度为0则默认顺时针旋转
#                             # 角速度大于0，即逆时针旋转中
#                             if bot_1['angular_velocity'] > 0:
#                                 command['rotate'][bot_id_1] = min_rotate_velocity + angular_velocity_kp * (
#                                         collision_area_left_boundary_angle - bot_1_relative_velocity_angle)
#                             # 角速度小于0，即顺时针旋转中
#                             elif bot_1['angular_velocity'] < 0:
#                                 command['rotate'][bot_id_1] = min_rotate_velocity + angular_velocity_kp * (
#                                         collision_area_right_boundary_angle - bot_1_relative_velocity_angle)
#                             # 角速度为0，默认顺时针旋转
#                             else:
#                                 command['rotate'][bot_id_1] = min_rotate_velocity + angular_velocity_kp * (
#                                         collision_area_right_boundary_angle - bot_1_relative_velocity_angle)
#                         # 该机器人已有避让机动
#                         break


def anti_collision(min_distance_kp, max_distance_kp):
    """
    简单的单目标防碰撞。
    :param min_distance_kp: 最小避障距离系数，距离大于圆心距乘该系数才触发避障
    :param max_distance_kp: 最大避障距离系数，距离小于圆心距乘该系数才触发避障
    :return: none
    """
    global bot_list, command
    angular_velocity_kp = 100  # 纠正角度偏差的比例系数
    collision_area_half_angle_kp = 1.2  # 增大碰撞区的系数
    # 单目标防碰撞，遍历每个主动避让机器人
    for bot_id_1, bot_1 in enumerate(bot_list):
        distance_list = []  # 机器人距离列表
        # 遍历其他三个机器人并计算与主动避让机器人的距离（但是为了保证下标一致也算上自身）
        for bot_id_2 in range(bot_num):
            # 计算距离
            distance = calculate_distance_between_bot(bot_id_1, bot_id_2)
            distance_list.append(distance)
        # 得出由近到远的索引数组
        distance_sort_index = numpy.argsort(distance_list)
        # 按由近到远的顺序遍历其他三个机器人
        for bot_id_2 in distance_sort_index:
            # 不算自己
            if bot_id_2 != bot_id_1:
                bot_2 = bot_list[bot_id_2]
                # 获取机器人半径
                if bot_1['object'] == 0:
                    bot_1_radius = no_load_radius
                else:
                    bot_1_radius = on_load_radius
                if bot_2['object'] == 0:
                    bot_2_radius = no_load_radius
                else:
                    bot_2_radius = on_load_radius
                # 触发避障的距离
                if (bot_1_radius + bot_2_radius) * min_distance_kp < distance_list[bot_id_2] < (
                        bot_1_radius + bot_2_radius) * max_distance_kp:
                    # 计算碰撞区
                    collision_area_half_angle = numpy.arcsin(
                        (bot_1_radius + bot_2_radius) / distance_list[bot_id_2])  # 碰撞区半角
                    # 适当增加碰撞区半角，但不超出最大值
                    if collision_area_half_angle * collision_area_half_angle_kp > numpy.pi / 2:
                        collision_area_half_angle = numpy.pi / 2
                    else:
                        collision_area_half_angle *= collision_area_half_angle_kp
                    collision_area_center_angle = calculate_angle_between_bot(bot_id_1, bot_id_2)  # 碰撞区中心方向角
                    collision_area_left_boundary_angle = collision_area_center_angle + collision_area_half_angle  # 碰撞区左边界方向角
                    collision_area_right_boundary_angle = collision_area_center_angle - collision_area_half_angle  # 碰撞区右边界角
                    # 计算机器人1相对于机器人2的速度
                    bot_1_relative_velocity_x = bot_1['linear_velocity'][0] - bot_2['linear_velocity'][0]  # x方向相对速度
                    bot_1_relative_velocity_y = bot_1['linear_velocity'][1] - bot_2['linear_velocity'][1]  # y方向相对速度
                    bot_1_relative_velocity_angle = numpy.arctan2(bot_1_relative_velocity_y,
                                                                  bot_1_relative_velocity_x)  # 相对速度方向
                    # 如果相对速度方向和碰撞区中心方向角之差大于pi（例如碰撞区是负角度，但相对速度方向在其+2pi后的正角度区间内）
                    if numpy.abs(bot_1_relative_velocity_angle - collision_area_center_angle) > numpy.pi:
                        # 变换到同号区间
                        if bot_1_relative_velocity_angle < 0:
                            bot_1_relative_velocity_angle += 2 * numpy.pi
                        elif bot_1_relative_velocity_angle > 0:
                            bot_1_relative_velocity_angle -= 2 * numpy.pi
                    # 如果相对速度方向在碰撞区内则会发生碰撞（恰巧在边界会相切，所以也避免），需要避让
                    if collision_area_right_boundary_angle < bot_1_relative_velocity_angle < collision_area_left_boundary_angle:
                        # 检测离哪边边界近就往哪边转，直接和中心方向角比较即可
                        # 离右边界近
                        if bot_1_relative_velocity_angle < collision_area_center_angle:
                            command['rotate'][bot_id_1] = min_rotate_velocity + angular_velocity_kp * (
                                    collision_area_right_boundary_angle - bot_1_relative_velocity_angle)
                        # 离左边界近
                        elif bot_1_relative_velocity_angle > collision_area_center_angle:
                            command['rotate'][bot_id_1] = min_rotate_velocity + angular_velocity_kp * (
                                    collision_area_left_boundary_angle - bot_1_relative_velocity_angle)
                        # 恰好位于中间
                        else:
                            # 查看目前的角速度方向，往当前已有的角速度方向转，角速度为0则默认顺时针旋转
                            # 角速度大于0，即逆时针旋转中
                            if bot_1['angular_velocity'] > 0:
                                command['rotate'][bot_id_1] = min_rotate_velocity + angular_velocity_kp * (
                                        collision_area_left_boundary_angle - bot_1_relative_velocity_angle)
                            # 角速度小于0，即顺时针旋转中
                            elif bot_1['angular_velocity'] < 0:
                                command['rotate'][bot_id_1] = min_rotate_velocity + angular_velocity_kp * (
                                        collision_area_right_boundary_angle - bot_1_relative_velocity_angle)
                            # 角速度为0，默认顺时针旋转
                            else:
                                command['rotate'][bot_id_1] = min_rotate_velocity + angular_velocity_kp * (
                                        collision_area_right_boundary_angle - bot_1_relative_velocity_angle)
                        # 该机器人已有避让机动
                        break


# endregion


# region 任务规划与分配
# 采用单段距离最短优先的算法，已淘汰，并且可能存在奇怪的bug，机器人会偶尔罢工
# def task_schedule_in_the_same_priority(bot_id, buy_table_type_list):
#     """
#     从同等优先级的工作台购买产品并出售的任务规划。
#     :param bot_id: 机器人ID
#     :param buy_table_type_list: 该优先级的工作台类型
#     :return: 0：不做规划，1：做出规划
#     """
#     global bot_list, table_list, table_type
#     min_distance = 100
#     min_distance_table_id_1 = -1
#     # 遍历该优先级的各种工作台
#     for table_type_id in buy_table_type_list:
#         # 如果存在这种工作台
#         if table_type_id in table_type:
#             # 遍历这种每个工作台
#             for table_id in table_index[f'type_{table_type_id}']:
#                 # 如果有产品未被分配取走任务
#                 if table_list[table_id]['projected_production'] == 1:
#                     distance = calculate_distance(bot_id, table_id)  # 计算机器人与该工作台的距离
#                     # 如果比目前存的最小距离小
#                     if distance < min_distance:
#                         min_distance = distance  # 更新最小距离
#                         min_distance_table_id_1 = table_id  # 更新最近工作台ID
#     # 如果找不到需要购买的产品
#     if min_distance_table_id_1 == -1:
#         return 0  # 不做规划
#     # 确定要去购买产品的工作台
#     temp_target_table = [min_distance_table_id_1]  # 储存要去购买产品的工作台
#     # 寻找离购买产品的工作台最近的可以出售该物品的工作台
#     object_type = table_list[min_distance_table_id_1]['type']  # 物品种类
#     if object_type == 1:
#         sell_table_type_list = [4, 5, 9]
#     elif object_type == 2:
#         sell_table_type_list = [4, 6, 9]
#     elif object_type == 3:
#         sell_table_type_list = [5, 6, 9]
#     elif object_type in [4, 5, 6]:
#         sell_table_type_list = [7, 9]
#     elif object_type == 7:
#         sell_table_type_list = [8, 9]
#     else:
#         sell_table_type_list = []
#     min_distance = 100
#     min_distance_table_id_2 = -1
#     # 遍历可以出售该物品的各种工作台
#     for table_type_id in sell_table_type_list:
#         # 如果存在这种工作台
#         if table_type_id in table_type:
#             # 如果是8、9，不用判断原材料格
#             if table_type_id in [8, 9]:
#                 # 遍历这种每个工作台
#                 for table_id in table_index[f'type_{table_type_id}']:
#                     distance = calculate_distance_between_table(temp_target_table[0],
#                                                                 table_id)  # 计算购买工作台与出售工作台之间的距离
#                     # 如果比目前存的最小距离小
#                     if distance < min_distance:
#                         min_distance = distance  # 更新最小距离
#                         min_distance_table_id_2 = table_id  # 更新最近工作台ID
#             # 如果不是8、9，要判断计划原材料格状态
#             else:
#                 # 遍历这种每个工作台
#                 for table_id in table_index[f'type_{table_type_id}']:
#                     # 如果缺少这个材料
#                     if object_type in table_list[table_id]['projected_missing_material']:
#                         distance = calculate_distance_between_table(temp_target_table[0],
#                                                                     table_id)  # 计算购买工作台与出售工作台之间的距离
#                         # 如果比目前存的最小距离小
#                         if distance < min_distance:
#                             min_distance = distance  # 更新最小距离
#                             min_distance_table_id_2 = table_id  # 更新最近工作台ID
#     # 找到了要购买的产品但是没地方卖
#     if min_distance_table_id_2 == -1:
#         return 0  # 不做规划
#     # 确定要去出售产品的工作台
#     temp_target_table.append(min_distance_table_id_2)  # 储存要去出售产品的工作台
#     table_list[min_distance_table_id_1]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
#     # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
#     if table_list[min_distance_table_id_2]['type'] not in [8, 9]:
#         table_list[min_distance_table_id_2]['projected_missing_material'].remove(object_type)  # 更新工作台的考虑计划的原料格状态
#     bot_list[bot_id]['target_table_id'] = tuple(temp_target_table)  # 把任务写到机器人字典中
#     bot_list[bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
#     return 1  # 做出规划


# 这一块所有的出售都改为1、2、3不能到9出售，这个改动主要是针对正式赛的地图
# def task_schedule_in_the_same_priority(bot_id: int, buy_table_type_list):
#     """
#     从同等优先级的工作台购买产品并出售的任务规划。
#     :param bot_id: 机器人ID
#     :param buy_table_type_list: 该优先级的工作台类型
#     :return: 0：不做规划，1：做出规划
#     """
#     global bot_list, table_list, table_type, table_index
#     min_distance_sum = 200
#     min_distance_table_id_1 = -1
#     min_distance_table_id_2 = -1
#     min_distance_object_type = 0
#     # 遍历该优先级的各种工作台
#     for table_type_id_1 in buy_table_type_list:
#         # 如果存在这种工作台
#         if table_type_id_1 in table_type:
#             # 遍历这种每个工作台
#             for table_id_1 in table_index[f'type_{table_type_id_1}']:
#                 # 如果有产品未被分配取走任务
#                 if table_list[table_id_1]['projected_production'] == 1:
#                     # 预检查是否需要通过优先级分配该购买任务
#                     # 如果只由原地购买承担
#                     if task_schedule_by_inplace_buy_precheck(table_id_1):
#                         continue  # 不去这个工作台购买
#                     # 如果需要由优先级承担
#                     else:
#                         distance_1 = calculate_distance(bot_id, table_id_1)  # 计算机器人与该工作台的距离
#                         # 寻找离本次选择的购买产品的工作台最近的可以出售该物品的工作台
#                         object_type = table_list[table_id_1]['type']  # 物品种类
#                         if object_type == 1:
#                             sell_table_type_list = [4, 5, 9]
#                         elif object_type == 2:
#                             sell_table_type_list = [4, 6, 9]
#                         elif object_type == 3:
#                             sell_table_type_list = [5, 6, 9]
#                         elif object_type in [4, 5, 6]:
#                             sell_table_type_list = [7, 9]
#                         elif object_type == 7:
#                             sell_table_type_list = [8, 9]
#                         else:
#                             sell_table_type_list = []
#                         # 遍历可以出售该物品的各种工作台
#                         for table_type_id_2 in sell_table_type_list:
#                             # 如果存在这种工作台
#                             if table_type_id_2 in table_type:
#                                 # 如果是8、9，不用判断原材料格
#                                 if table_type_id_2 in [8, 9]:
#                                     # 遍历这种每个工作台
#                                     for table_id_2 in table_index[f'type_{table_type_id_2}']:
#                                         distance_2 = calculate_distance_between_table(table_id_1,
#                                                                                       table_id_2)  # 计算购买工作台与出售工作台之间的距离
#                                         distance_sum = distance_1 + distance_2  # 计算两段距离和
#                                         # 如果当前距离和小于存的最小距离和
#                                         if distance_sum < min_distance_sum:
#                                             min_distance_sum = distance_sum  # 更新最小距离和
#                                             min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
#                                             min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
#                                             min_distance_object_type = object_type  # 更新最小距离和下的物品类型
#                                 # 如果不是8、9，要判断计划原材料格状态
#                                 else:
#                                     # 遍历这种每个工作台
#                                     for table_id_2 in table_index[f'type_{table_type_id_2}']:
#                                         # 如果缺少这个材料
#                                         if object_type in table_list[table_id_2]['projected_missing_material']:
#                                             distance_2 = calculate_distance_between_table(table_id_1,
#                                                                                           table_id_2)  # 计算购买工作台与出售工作台之间的距离
#                                             distance_sum = distance_1 + distance_2  # 计算两段距离和
#                                             # 如果当前距离和小于存的最小距离和
#                                             if distance_sum < min_distance_sum:
#                                                 min_distance_sum = distance_sum  # 更新最小距离和
#                                                 min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
#                                                 min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
#                                                 min_distance_object_type = object_type  # 更新最小距离和下的物品类型
#     # 如果找不到需要购买的产品或找到了要购买的产品但是没地方卖
#     if min_distance_table_id_1 == -1 or min_distance_table_id_2 == -1:
#         return 0  # 不做规划
#     else:
#         table_list[min_distance_table_id_1]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
#         # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
#         if table_list[min_distance_table_id_2]['type'] not in [8, 9]:
#             table_list[min_distance_table_id_2]['projected_missing_material'].remove(
#                 min_distance_object_type)  # 更新工作台的考虑计划的原料格状态
#         bot_list[bot_id]['target_table_id'] = (min_distance_table_id_1, min_distance_table_id_2)  # 把任务写到机器人字典中
#         bot_list[bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
#         return 1  # 做出规划
#
#
# def task_schedule_by_inplace_buy(bot_id: int):
#     """
#     机器人出售后如果工作台有产品的话直接原地购买，若存在抢占情况则把被抢占该购买任务的机器人的工作状态改为空闲同时停止运动。
#     :param bot_id: 机器人ID
#     :return: 0：不做规划，1：做出规划
#     """
#     global bot_list, table_list
#     # 等待时间阈值
#     wait_threshold = 10
#     # 如果机器人在工作台处时空闲
#     if bot_list[bot_id]['table_id'] != -1:
#         inplace_buy_table_id = bot_list[bot_id]['table_id']  # 所在工作台ID
#         # 如果是4/5/6/7类工作台（因为不可能在1/2/3类工作台空闲，在8/9则是不可能有产品）
#         if table_list[inplace_buy_table_id]['type'] in [4, 5, 6, 7]:
#             # 如果这个工作台有产品（非计划）
#             if table_list[inplace_buy_table_id]['production'] == 1:
#                 # 即使有预检查也可能存在购买抢占的情况，遍历其他机器人看看有没有要买这个工作台产品的任务
#                 for bot_id_2 in range(bot_num):
#                     # 如果不是当前机器人
#                     if bot_id_2 != bot_id:
#                         # 如果其购买任务被抢占
#                         if bot_list[bot_id_2]['target_table_id'][0] == inplace_buy_table_id and bot_list[bot_id_2][
#                             'work_status'] == 1:
#                             # 停止运动
#                             command['forward'][bot_id_2] = 0
#                             command['rotate'][bot_id_2] = 0
#                             # 修改运动状态为空闲
#                             bot_list[bot_id_2]['work_status'] = 0
#                             # 当前机器人直接购买并设置出售目标工作台为被抢占机器人的出售目标工作台
#                             bot_list[bot_id]['target_table_id'] = (
#                                 inplace_buy_table_id, bot_list[bot_id_2]['target_table_id'][1])  # 把任务写到当前机器人种
#                             bot_list[bot_id]['work_status'] = 1  # 当前机器人运动状态改为前往第一个目标工作台
#                             return 1  # 做出规划
#                 # 没有机器人的任务被抢占
#                 # 寻找最近的可以出售该物品的工作台
#                 object_type = table_list[inplace_buy_table_id]['type']  # 物品种类
#                 if object_type in [4, 5, 6]:
#                     sell_table_type_list = [7, 9]
#                 elif object_type == 7:
#                     sell_table_type_list = [8, 9]
#                 else:
#                     sell_table_type_list = []
#                 min_distance = 100
#                 min_distance_table_id = -1
#                 # 遍历可以出售该物品的各种工作台
#                 for table_type_id in sell_table_type_list:
#                     # 如果存在这种工作台
#                     if table_type_id in table_type:
#                         # 如果是8、9，不用判断原材料格
#                         if table_type_id in [8, 9]:
#                             # 遍历这种每个工作台
#                             for table_id in table_index[f'type_{table_type_id}']:
#                                 distance = calculate_distance_between_table(inplace_buy_table_id,
#                                                                             table_id)  # 计算购买工作台与出售工作台之间的距离
#                                 # 如果比目前存的最小距离小
#                                 if distance < min_distance:
#                                     min_distance = distance  # 更新最小距离
#                                     min_distance_table_id = table_id  # 更新最近工作台ID
#                         # 如果不是8、9，要判断计划原材料格状态
#                         else:
#                             # 遍历这种每个工作台
#                             for table_id in table_index[f'type_{table_type_id}']:
#                                 # 如果缺少这个材料
#                                 if object_type in table_list[table_id]['projected_missing_material']:
#                                     distance = calculate_distance_between_table(inplace_buy_table_id,
#                                                                                 table_id)  # 计算购买工作台与出售工作台之间的距离
#                                     # 如果比目前存的最小距离小
#                                     if distance < min_distance:
#                                         min_distance = distance  # 更新最小距离
#                                         min_distance_table_id = table_id  # 更新最近工作台ID
#                 # 没地方卖
#                 if min_distance_table_id == -1:
#                     return 0  # 不做规划
#                 else:
#                     table_list[inplace_buy_table_id]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
#                     # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
#                     if table_list[min_distance_table_id]['type'] not in [8, 9]:
#                         table_list[min_distance_table_id]['projected_missing_material'].remove(
#                             object_type)  # 更新工作台的考虑计划的原料格状态
#                     # 修改机器人任务
#                     bot_list[bot_id]['target_table_id'] = (
#                         inplace_buy_table_id, min_distance_table_id)  # 把任务写到机器人字典中
#                     bot_list[bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
#                     return 1  # 做出规划
#             # 如果这个工作台没产品
#             else:
#                 # 如果剩余时间在等待阈值内
#                 if -1 < table_list[inplace_buy_table_id]['remaining_time'] < wait_threshold:
#                     return 1  # 等待，假装做出规划
#                 return 0  # 不做规划
#
#
# # 即使有预检查也可能发生抢占的情况
# def task_schedule_by_inplace_buy_precheck(table_id: int):
#     """
#     机器人在出售工作台直接购买产品的预检查，用于决定该购买任务是否由task_schedule_by_inplace_buy()承担。
#     :param table_id: 需要进行检查的工作台ID
#     :return: 0：由优先级承担（承担两个产品之一或没有原地购买条件），1：由出售后原地购买承担
#     """
#     global bot_list, table_list
#     # 判断是否是可以进行出售后原地购买的工作台类型
#     if table_list[table_id]['type'] in [4, 5, 6, 7]:
#         # 遍历机器人
#         for bot in bot_list:
#             # 如果有机器人的出售目标工作台是这个工作台并且是工作状态
#             if bot['target_table_id'][1] == table_id and bot['work_status'] != 0:
#                 # 如果只有一个产品（也就是没有产品堵塞）
#                 # 不检查产品格情况，需要在确定有产品时才能调用
#                 if table_list[table_id]['remaining_time'] != 0:
#                     return 1  # 由出售后原地购买算法承担
#                 # 如果有产品堵塞，则可由优先级分发一个购买任务
#                 else:
#                     return 0  # 由优先级承担
#         # 没有机器人的出售目标是这个工作台
#         return 0  # 由优先级承担
#     # 类型不符合
#     else:
#         return 0  # 由优先级承担
#
#
# def task_schedule_in_preempting_priority(buy_table_type_list):
#     """
#     抢占优先级任务规划，每次执行最多发生一个抢占任务。
#     :param buy_table_type_list: 抢占优先级购买工作台类型列表
#     :return: 0：不做抢占，1：做出抢占
#     """
#     global bot_list, table_list
#     preemption_distance_kp = 0.6  # 抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
#     min_distance_sum = 200
#     min_distance_table_id_1 = -1
#     min_distance_table_id_2 = -1
#     min_distance_object_type = 0
#     min_distance_bot_id = -1
#     # 先看有没有可以被抢占的机器人
#     for bot in bot_list:
#         # 如果有可以被抢占123的机器人
#         if bot['work_status'] == 1 and bot['object'] == 0 and table_list[bot['target_table_id'][0]]['type'] in [1, 2,
#                                                                                                                 3]:
#             preemption_buy_table_type_list = [1, 2, 3]
#             break
#     # 没有可以被抢占123的机器人
#     else:
#         # 如果抢占任务是7
#         if buy_table_type_list == [7]:
#             # 再找
#             for bot in bot_list:
#                 # 如果有可以被抢占456的机器人也行
#                 if bot['work_status'] == 1 and bot['object'] == 0 and table_list[bot['target_table_id'][0]]['type'] in [
#                     4, 5, 6]:
#                     preemption_buy_table_type_list = [4, 5, 6]
#                     break
#             # 456也没有
#             else:
#                 return 0
#         # 不是7
#         else:
#             return 0
#     # 遍历该抢占优先级的各种工作台
#     for table_type_id_1 in buy_table_type_list:
#         # 如果存在这种工作台
#         if table_type_id_1 in table_type:
#             # 遍历这种每个工作台
#             for table_id_1 in table_index[f'type_{table_type_id_1}']:
#                 # 如果有产品未被分配取走任务
#                 if table_list[table_id_1]['projected_production'] == 1:
#                     # 预检查是否需要通过优先级分配该购买任务
#                     # 如果只由原地购买承担
#                     if task_schedule_by_inplace_buy_precheck(table_id_1):
#                         continue  # 不去这个工作台购买
#                     # 如果需要由分析是否需要抢占优先级承担
#                     else:
#                         # 遍历机器人
#                         for bot_id in range(bot_num):
#                             # 抢占任务
#                             if bot_list[bot_id]['work_status'] == 1 and bot_list[bot_id]['object'] == 0 and \
#                                     table_list[bot_list[bot_id]['target_table_id'][0]][
#                                         'type'] in preemption_buy_table_type_list:
#                                 remaining_distance = calculate_distance(bot_id, bot_list[bot_id]['target_table_id'][
#                                     0])  # 计算该机器人与原本的购买工作台之间剩余的距离
#                                 preemption_distance = calculate_distance(bot_id, table_id_1)  # 计算该机器人与抢占任务购买工作台之间的距离
#                                 # 如果剩余距离较长
#                                 if remaining_distance >= preemption_distance_kp * preemption_distance:
#                                     # 可以抢占
#                                     # 寻找离本次选择的购买产品的工作台最近的可以出售该物品的工作台
#                                     distance_1 = preemption_distance
#                                     object_type = table_list[table_id_1]['type']  # 物品种类
#                                     if object_type == 1:
#                                         sell_table_type_list = [4, 5, 9]
#                                     elif object_type == 2:
#                                         sell_table_type_list = [4, 6, 9]
#                                     elif object_type == 3:
#                                         sell_table_type_list = [5, 6, 9]
#                                     elif object_type in [4, 5, 6]:
#                                         sell_table_type_list = [7, 9]
#                                     elif object_type == 7:
#                                         sell_table_type_list = [8, 9]
#                                     else:
#                                         sell_table_type_list = []
#                                     # 遍历可以出售该物品的各种工作台
#                                     for table_type_id_2 in sell_table_type_list:
#                                         # 如果存在这种工作台
#                                         if table_type_id_2 in table_type:
#                                             # 如果是8、9，不用判断原材料格
#                                             if table_type_id_2 in [8, 9]:
#                                                 # 遍历这种每个工作台
#                                                 for table_id_2 in table_index[f'type_{table_type_id_2}']:
#                                                     distance_2 = calculate_distance_between_table(table_id_1,
#                                                                                                   table_id_2)  # 计算购买工作台与出售工作台之间的距离
#                                                     distance_sum = distance_1 + distance_2  # 计算两段距离和
#                                                     # 如果当前距离和小于存的最小距离和
#                                                     if distance_sum < min_distance_sum:
#                                                         min_distance_sum = distance_sum  # 更新最小距离和
#                                                         min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
#                                                         min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
#                                                         min_distance_bot_id = bot_id  # 更新最小距离和下的机器人ID
#                                                         min_distance_object_type = object_type  # 更新最小距离和下的物品类型
#                                             # 如果不是8、9，要判断计划原材料格状态
#                                             else:
#                                                 # 遍历这种每个工作台
#                                                 for table_id_2 in table_index[f'type_{table_type_id_2}']:
#                                                     # 如果缺少这个材料
#                                                     if object_type in table_list[table_id_2][
#                                                         'projected_missing_material']:
#                                                         distance_2 = calculate_distance_between_table(table_id_1,
#                                                                                                       table_id_2)  # 计算购买工作台与出售工作台之间的距离
#                                                         distance_sum = distance_1 + distance_2  # 计算两段距离和
#                                                         # 如果当前距离和小于存的最小距离和
#                                                         if distance_sum < min_distance_sum:
#                                                             min_distance_sum = distance_sum  # 更新最小距离和
#                                                             min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
#                                                             min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
#                                                             min_distance_bot_id = bot_id  # 更新最小距离和下的机器人ID
#                                                             min_distance_object_type = object_type  # 更新最小距离和下的物品类型
#     # 如果找不到需要购买的产品或找到了要购买的产品但是没地方卖
#     if min_distance_table_id_1 == -1 or min_distance_table_id_2 == -1:
#         return 0  # 不做抢占
#     # 发生抢占
#     else:
#         # 复原原计划购买工作台的计划产品格状态
#         table_list[bot_list[min_distance_bot_id]['target_table_id'][0]]['projected_production'] = 1
#         # 复原原计划出售工作台的计划原材料格状态
#         table_list[bot_list[min_distance_bot_id]['target_table_id'][1]]['projected_missing_material'].append(
#             table_list[bot_list[min_distance_bot_id]['target_table_id'][0]]['type'])
#         # 新任务
#         table_list[min_distance_table_id_1]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
#         # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
#         if table_list[min_distance_table_id_2]['type'] not in [8, 9]:
#             table_list[min_distance_table_id_2]['projected_missing_material'].remove(
#                 min_distance_object_type)  # 更新工作台的考虑计划的原料格状态
#         bot_list[min_distance_bot_id]['target_table_id'] = (
#             min_distance_table_id_1, min_distance_table_id_2)  # 把任务写到机器人字典中
#         bot_list[min_distance_bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
#         logging.info('1')
#         return 1  # 做出抢占


def task_schedule_in_the_same_priority(bot_id: int, buy_table_type_list):
    """
    从同等优先级的工作台购买产品并出售的任务规划。
    :param bot_id: 机器人ID
    :param buy_table_type_list: 该优先级的工作台类型
    :return: 0：不做规划，1：做出规划
    """
    global bot_list, table_list, table_type, table_index
    min_distance_sum = 200
    min_distance_table_id_1 = -1
    min_distance_table_id_2 = -1
    min_distance_object_type = 0
    # 遍历该优先级的各种工作台
    for table_type_id_1 in buy_table_type_list:
        # 如果存在这种工作台
        if table_type_id_1 in table_type:
            # 遍历这种每个工作台
            for table_id_1 in table_index[f'type_{table_type_id_1}']:
                # 如果有产品未被分配取走任务
                if table_list[table_id_1]['projected_production'] == 1:
                    # 预检查是否需要通过优先级分配该购买任务
                    # 如果只由原地购买承担
                    if task_schedule_by_inplace_buy_precheck(table_id_1):
                        continue  # 不去这个工作台购买
                    # 如果需要由优先级承担
                    else:
                        distance_1 = calculate_distance(bot_id, table_id_1)  # 计算机器人与该工作台的距离
                        # 寻找离本次选择的购买产品的工作台最近的可以出售该物品的工作台（1、2、3不到9卖）
                        object_type = table_list[table_id_1]['type']  # 物品种类
                        if object_type == 1:
                            sell_table_type_list = [4, 5]
                        elif object_type == 2:
                            sell_table_type_list = [4, 6]
                        elif object_type == 3:
                            sell_table_type_list = [5, 6]
                        elif object_type in [4, 5, 6]:
                            sell_table_type_list = [7, 9]
                        elif object_type == 7:
                            sell_table_type_list = [8, 9]
                        else:
                            sell_table_type_list = []
                        # 遍历可以出售该物品的各种工作台
                        for table_type_id_2 in sell_table_type_list:
                            # 如果存在这种工作台
                            if table_type_id_2 in table_type:
                                # 如果是8、9，不用判断原材料格
                                if table_type_id_2 in [8, 9]:
                                    # 遍历这种每个工作台
                                    for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                        distance_2 = calculate_distance_between_table(table_id_1,
                                                                                      table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                        distance_sum = distance_1 + distance_2  # 计算两段距离和
                                        # 如果当前距离和小于存的最小距离和
                                        if distance_sum < min_distance_sum:
                                            min_distance_sum = distance_sum  # 更新最小距离和
                                            min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                            min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                            min_distance_object_type = object_type  # 更新最小距离和下的物品类型
                                # 如果不是8、9，要判断计划原材料格状态
                                else:
                                    # 遍历这种每个工作台
                                    for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                        # 如果缺少这个材料
                                        if object_type in table_list[table_id_2]['projected_missing_material']:
                                            distance_2 = calculate_distance_between_table(table_id_1,
                                                                                          table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                            distance_sum = distance_1 + distance_2  # 计算两段距离和
                                            # 如果当前距离和小于存的最小距离和
                                            if distance_sum < min_distance_sum:
                                                min_distance_sum = distance_sum  # 更新最小距离和
                                                min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                                min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                                min_distance_object_type = object_type  # 更新最小距离和下的物品类型
    # 如果找不到需要购买的产品或找到了要购买的产品但是没地方卖
    if min_distance_table_id_1 == -1 or min_distance_table_id_2 == -1:
        return 0  # 不做规划
    else:
        table_list[min_distance_table_id_1]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
        # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
        if table_list[min_distance_table_id_2]['type'] not in [8, 9]:
            table_list[min_distance_table_id_2]['projected_missing_material'].remove(
                min_distance_object_type)  # 更新工作台的考虑计划的原料格状态
        bot_list[bot_id]['target_table_id'] = (min_distance_table_id_1, min_distance_table_id_2)  # 把任务写到机器人字典中
        bot_list[bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
        return 1  # 做出规划


# 等待时间改为传参
# def task_schedule_by_inplace_buy(bot_id: int):
#     """
#     机器人出售后如果工作台有产品的话直接原地购买，若存在抢占情况则把被抢占该购买任务的机器人的工作状态改为空闲同时停止运动。
#     :param bot_id: 机器人ID
#     :return: 0：不做规划，1：做出规划
#     """
#     global bot_list, table_list
#     # 等待时间阈值
#     wait_threshold = 10
#     # 如果机器人在工作台处时空闲
#     if bot_list[bot_id]['table_id'] != -1:
#         inplace_buy_table_id = bot_list[bot_id]['table_id']  # 所在工作台ID
#         # 如果是4/5/6/7类工作台（因为不可能在1/2/3类工作台空闲，在8/9则是不可能有产品）
#         if table_list[inplace_buy_table_id]['type'] in [4, 5, 6, 7]:
#             # 如果这个工作台有产品（非计划）
#             if table_list[inplace_buy_table_id]['production'] == 1:
#                 # 即使有预检查也可能存在购买抢占的情况，遍历其他机器人看看有没有要买这个工作台产品的任务
#                 for bot_id_2 in range(bot_num):
#                     # 如果不是当前机器人
#                     if bot_id_2 != bot_id:
#                         # 如果其购买任务被抢占
#                         if bot_list[bot_id_2]['target_table_id'][0] == inplace_buy_table_id and bot_list[bot_id_2][
#                             'work_status'] == 1:
#                             # 停止运动
#                             command['forward'][bot_id_2] = 0
#                             command['rotate'][bot_id_2] = 0
#                             # 修改运动状态为空闲
#                             bot_list[bot_id_2]['work_status'] = 0
#                             # 当前机器人直接购买并设置出售目标工作台为被抢占机器人的出售目标工作台
#                             bot_list[bot_id]['target_table_id'] = (
#                                 inplace_buy_table_id, bot_list[bot_id_2]['target_table_id'][1])  # 把任务写到当前机器人种
#                             bot_list[bot_id]['work_status'] = 1  # 当前机器人运动状态改为前往第一个目标工作台
#                             return 1  # 做出规划
#                 # 没有机器人的任务被抢占
#                 # 寻找最近的可以出售该物品的工作台
#                 object_type = table_list[inplace_buy_table_id]['type']  # 物品种类
#                 if object_type in [4, 5, 6]:
#                     sell_table_type_list = [7, 9]
#                 elif object_type == 7:
#                     sell_table_type_list = [8, 9]
#                 else:
#                     sell_table_type_list = []
#                 min_distance = 100
#                 min_distance_table_id = -1
#                 # 遍历可以出售该物品的各种工作台
#                 for table_type_id in sell_table_type_list:
#                     # 如果存在这种工作台
#                     if table_type_id in table_type:
#                         # 如果是8、9，不用判断原材料格
#                         if table_type_id in [8, 9]:
#                             # 遍历这种每个工作台
#                             for table_id in table_index[f'type_{table_type_id}']:
#                                 distance = calculate_distance_between_table(inplace_buy_table_id,
#                                                                             table_id)  # 计算购买工作台与出售工作台之间的距离
#                                 # 如果比目前存的最小距离小
#                                 if distance < min_distance:
#                                     min_distance = distance  # 更新最小距离
#                                     min_distance_table_id = table_id  # 更新最近工作台ID
#                         # 如果不是8、9，要判断计划原材料格状态
#                         else:
#                             # 遍历这种每个工作台
#                             for table_id in table_index[f'type_{table_type_id}']:
#                                 # 如果缺少这个材料
#                                 if object_type in table_list[table_id]['projected_missing_material']:
#                                     distance = calculate_distance_between_table(inplace_buy_table_id,
#                                                                                 table_id)  # 计算购买工作台与出售工作台之间的距离
#                                     # 如果比目前存的最小距离小
#                                     if distance < min_distance:
#                                         min_distance = distance  # 更新最小距离
#                                         min_distance_table_id = table_id  # 更新最近工作台ID
#                 # 没地方卖
#                 if min_distance_table_id == -1:
#                     return 0  # 不做规划
#                 else:
#                     table_list[inplace_buy_table_id]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
#                     # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
#                     if table_list[min_distance_table_id]['type'] not in [8, 9]:
#                         table_list[min_distance_table_id]['projected_missing_material'].remove(
#                             object_type)  # 更新工作台的考虑计划的原料格状态
#                     # 修改机器人任务
#                     bot_list[bot_id]['target_table_id'] = (
#                         inplace_buy_table_id, min_distance_table_id)  # 把任务写到机器人字典中
#                     bot_list[bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
#                     return 1  # 做出规划
#             # 如果这个工作台没产品
#             else:
#                 # 如果剩余时间在等待阈值内
#                 if -1 < table_list[inplace_buy_table_id]['remaining_time'] < wait_threshold:
#                     return 1  # 等待，假装做出规划
#                 return 0  # 不做规划


def task_schedule_by_inplace_buy(bot_id: int, wait_threshold):
    """
    机器人出售后如果工作台有产品的话直接原地购买，若存在抢占情况则把被抢占该购买任务的机器人的工作状态改为空闲同时停止运动。
    :param bot_id: 机器人ID
    :param wait_threshold: 最长等待时间
    :return: 0：不做规划，1：做出规划
    """
    global bot_list, table_list
    # 如果机器人在工作台处时空闲
    if bot_list[bot_id]['table_id'] != -1:
        inplace_buy_table_id = bot_list[bot_id]['table_id']  # 所在工作台ID
        # 如果是4/5/6/7类工作台（因为不可能在1/2/3类工作台空闲，在8/9则是不可能有产品）
        if table_list[inplace_buy_table_id]['type'] in [4, 5, 6, 7]:
            # 如果这个工作台有产品（非计划）
            if table_list[inplace_buy_table_id]['production'] == 1:
                # 即使有预检查也可能存在购买抢占的情况，遍历其他机器人看看有没有要买这个工作台产品的任务
                for bot_id_2 in range(bot_num):
                    # 如果不是当前机器人
                    if bot_id_2 != bot_id:
                        # 如果其购买任务被抢占
                        if bot_list[bot_id_2]['target_table_id'][0] == inplace_buy_table_id and bot_list[bot_id_2][
                            'work_status'] == 1:
                            # 停止运动
                            command['forward'][bot_id_2] = 0
                            command['rotate'][bot_id_2] = 0
                            # 修改运动状态为空闲
                            bot_list[bot_id_2]['work_status'] = 0
                            # 当前机器人直接购买并设置出售目标工作台为被抢占机器人的出售目标工作台
                            bot_list[bot_id]['target_table_id'] = (
                                inplace_buy_table_id, bot_list[bot_id_2]['target_table_id'][1])  # 把任务写到当前机器人种
                            bot_list[bot_id]['work_status'] = 1  # 当前机器人运动状态改为前往第一个目标工作台
                            return 1  # 做出规划
                # 没有机器人的任务被抢占
                # 寻找最近的可以出售该物品的工作台
                object_type = table_list[inplace_buy_table_id]['type']  # 物品种类
                if object_type in [4, 5, 6]:
                    sell_table_type_list = [7, 9]
                elif object_type == 7:
                    sell_table_type_list = [8, 9]
                else:
                    sell_table_type_list = []
                min_distance = 100
                min_distance_table_id = -1
                # 遍历可以出售该物品的各种工作台
                for table_type_id in sell_table_type_list:
                    # 如果存在这种工作台
                    if table_type_id in table_type:
                        # 如果是8、9，不用判断原材料格
                        if table_type_id in [8, 9]:
                            # 遍历这种每个工作台
                            for table_id in table_index[f'type_{table_type_id}']:
                                distance = calculate_distance_between_table(inplace_buy_table_id,
                                                                            table_id)  # 计算购买工作台与出售工作台之间的距离
                                # 如果比目前存的最小距离小
                                if distance < min_distance:
                                    min_distance = distance  # 更新最小距离
                                    min_distance_table_id = table_id  # 更新最近工作台ID
                        # 如果不是8、9，要判断计划原材料格状态
                        else:
                            # 遍历这种每个工作台
                            for table_id in table_index[f'type_{table_type_id}']:
                                # 如果缺少这个材料
                                if object_type in table_list[table_id]['projected_missing_material']:
                                    distance = calculate_distance_between_table(inplace_buy_table_id,
                                                                                table_id)  # 计算购买工作台与出售工作台之间的距离
                                    # 如果比目前存的最小距离小
                                    if distance < min_distance:
                                        min_distance = distance  # 更新最小距离
                                        min_distance_table_id = table_id  # 更新最近工作台ID
                # 没地方卖
                if min_distance_table_id == -1:
                    return 0  # 不做规划
                else:
                    table_list[inplace_buy_table_id]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
                    # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
                    if table_list[min_distance_table_id]['type'] not in [8, 9]:
                        table_list[min_distance_table_id]['projected_missing_material'].remove(
                            object_type)  # 更新工作台的考虑计划的原料格状态
                    # 修改机器人任务
                    bot_list[bot_id]['target_table_id'] = (
                        inplace_buy_table_id, min_distance_table_id)  # 把任务写到机器人字典中
                    bot_list[bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
                    return 1  # 做出规划
            # 如果这个工作台没产品
            else:
                # 如果剩余时间在等待阈值内
                if -1 < table_list[inplace_buy_table_id]['remaining_time'] < wait_threshold:
                    return 1  # 等待，假装做出规划
                return 0  # 不做规划


# 即使有预检查也可能发生抢占的情况
def task_schedule_by_inplace_buy_precheck(table_id: int):
    """
    机器人在出售工作台直接购买产品的预检查，用于决定该购买任务是否由task_schedule_by_inplace_buy()承担。
    :param table_id: 需要进行检查的工作台ID
    :return: 0：由优先级承担（承担两个产品之一或没有原地购买条件），1：由出售后原地购买承担
    """
    global bot_list, table_list
    # 判断是否是可以进行出售后原地购买的工作台类型
    if table_list[table_id]['type'] in [4, 5, 6, 7]:
        # 遍历机器人
        for bot in bot_list:
            # 如果有机器人的出售目标工作台是这个工作台并且是工作状态
            if bot['work_status'] != 0:
                if bot['target_table_id'][1] == table_id:
                    # 如果只有一个产品（也就是没有产品堵塞）
                    # 不检查产品格情况，需要在确定有产品时才能调用
                    if table_list[table_id]['remaining_time'] != 0:
                        return 1  # 由出售后原地购买算法承担
                    # 如果有产品堵塞，则可由优先级分发一个购买任务
                    else:
                        return 0  # 由优先级承担
        # 没有机器人的出售目标是这个工作台
        return 0  # 由优先级承担
    # 类型不符合
    else:
        return 0  # 由优先级承担


# 抢占距离系数改为传参
# def task_schedule_in_preempting_priority(buy_table_type_list):
#     """
#     抢占优先级任务规划，每次执行最多发生一个抢占任务。
#     :param buy_table_type_list: 抢占优先级购买工作台类型列表
#     :return: 0：不做抢占，1：做出抢占
#     """
#     global bot_list, table_list
#     preemption_distance_kp = 0.6  # 抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
#     min_distance_sum = 200
#     min_distance_table_id_1 = -1
#     min_distance_table_id_2 = -1
#     min_distance_object_type = 0
#     min_distance_bot_id = -1
#     # 先看有没有可以被抢占的机器人
#     for bot in bot_list:
#         # 如果有可以被抢占123的机器人
#         if bot['work_status'] == 1 and bot['object'] == 0 and table_list[bot['target_table_id'][0]]['type'] in [1, 2,
#                                                                                                                 3]:
#             preemption_buy_table_type_list = [1, 2, 3]
#             break
#     # 没有可以被抢占123的机器人
#     else:
#         # 如果抢占任务是7
#         if buy_table_type_list == [7]:
#             # 再找
#             for bot in bot_list:
#                 # 如果有可以被抢占456的机器人也行
#                 if bot['work_status'] == 1 and bot['object'] == 0 and table_list[bot['target_table_id'][0]]['type'] in [
#                     4, 5, 6]:
#                     preemption_buy_table_type_list = [4, 5, 6]
#                     break
#             # 456也没有
#             else:
#                 return 0
#         # 不是7
#         else:
#             return 0
#     # 遍历该抢占优先级的各种工作台
#     for table_type_id_1 in buy_table_type_list:
#         # 如果存在这种工作台
#         if table_type_id_1 in table_type:
#             # 遍历这种每个工作台
#             for table_id_1 in table_index[f'type_{table_type_id_1}']:
#                 # 如果有产品未被分配取走任务
#                 if table_list[table_id_1]['projected_production'] == 1:
#                     # 预检查是否需要通过优先级分配该购买任务
#                     # 如果只由原地购买承担
#                     if task_schedule_by_inplace_buy_precheck(table_id_1):
#                         continue  # 不去这个工作台购买
#                     # 如果需要由分析是否需要抢占优先级承担
#                     else:
#                         # 遍历机器人
#                         for bot_id in range(bot_num):
#                             # 抢占任务
#                             if bot_list[bot_id]['work_status'] == 1 and bot_list[bot_id]['object'] == 0 and \
#                                     table_list[bot_list[bot_id]['target_table_id'][0]][
#                                         'type'] in preemption_buy_table_type_list:
#                                 remaining_distance = calculate_distance(bot_id, bot_list[bot_id]['target_table_id'][
#                                     0])  # 计算该机器人与原本的购买工作台之间剩余的距离
#                                 preemption_distance = calculate_distance(bot_id, table_id_1)  # 计算该机器人与抢占任务购买工作台之间的距离
#                                 # 如果剩余距离较长
#                                 if remaining_distance >= preemption_distance_kp * preemption_distance:
#                                     # 可以抢占
#                                     # 寻找离本次选择的购买产品的工作台最近的可以出售该物品的工作台（1、2、3不到9卖）
#                                     distance_1 = preemption_distance
#                                     object_type = table_list[table_id_1]['type']  # 物品种类
#                                     if object_type == 1:
#                                         sell_table_type_list = [4, 5]
#                                     elif object_type == 2:
#                                         sell_table_type_list = [4, 6]
#                                     elif object_type == 3:
#                                         sell_table_type_list = [5, 6]
#                                     elif object_type in [4, 5, 6]:
#                                         sell_table_type_list = [7, 9]
#                                     elif object_type == 7:
#                                         sell_table_type_list = [8, 9]
#                                     else:
#                                         sell_table_type_list = []
#                                     # 遍历可以出售该物品的各种工作台
#                                     for table_type_id_2 in sell_table_type_list:
#                                         # 如果存在这种工作台
#                                         if table_type_id_2 in table_type:
#                                             # 如果是8、9，不用判断原材料格
#                                             if table_type_id_2 in [8, 9]:
#                                                 # 遍历这种每个工作台
#                                                 for table_id_2 in table_index[f'type_{table_type_id_2}']:
#                                                     distance_2 = calculate_distance_between_table(table_id_1,
#                                                                                                   table_id_2)  # 计算购买工作台与出售工作台之间的距离
#                                                     distance_sum = distance_1 + distance_2  # 计算两段距离和
#                                                     # 如果当前距离和小于存的最小距离和
#                                                     if distance_sum < min_distance_sum:
#                                                         min_distance_sum = distance_sum  # 更新最小距离和
#                                                         min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
#                                                         min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
#                                                         min_distance_bot_id = bot_id  # 更新最小距离和下的机器人ID
#                                                         min_distance_object_type = object_type  # 更新最小距离和下的物品类型
#                                             # 如果不是8、9，要判断计划原材料格状态
#                                             else:
#                                                 # 遍历这种每个工作台
#                                                 for table_id_2 in table_index[f'type_{table_type_id_2}']:
#                                                     # 如果缺少这个材料
#                                                     if object_type in table_list[table_id_2][
#                                                         'projected_missing_material']:
#                                                         distance_2 = calculate_distance_between_table(table_id_1,
#                                                                                                       table_id_2)  # 计算购买工作台与出售工作台之间的距离
#                                                         distance_sum = distance_1 + distance_2  # 计算两段距离和
#                                                         # 如果当前距离和小于存的最小距离和
#                                                         if distance_sum < min_distance_sum:
#                                                             min_distance_sum = distance_sum  # 更新最小距离和
#                                                             min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
#                                                             min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
#                                                             min_distance_bot_id = bot_id  # 更新最小距离和下的机器人ID
#                                                             min_distance_object_type = object_type  # 更新最小距离和下的物品类型
#     # 如果找不到需要购买的产品或找到了要购买的产品但是没地方卖
#     if min_distance_table_id_1 == -1 or min_distance_table_id_2 == -1:
#         return 0  # 不做抢占
#     # 发生抢占
#     else:
#         # 复原原计划购买工作台的计划产品格状态
#         table_list[bot_list[min_distance_bot_id]['target_table_id'][0]]['projected_production'] = 1
#         # 复原原计划出售工作台的计划原材料格状态
#         table_list[bot_list[min_distance_bot_id]['target_table_id'][1]]['projected_missing_material'].append(
#             table_list[bot_list[min_distance_bot_id]['target_table_id'][0]]['type'])
#         # 新任务
#         table_list[min_distance_table_id_1]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
#         # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
#         if table_list[min_distance_table_id_2]['type'] not in [8, 9]:
#             table_list[min_distance_table_id_2]['projected_missing_material'].remove(
#                 min_distance_object_type)  # 更新工作台的考虑计划的原料格状态
#         bot_list[min_distance_bot_id]['target_table_id'] = (
#             min_distance_table_id_1, min_distance_table_id_2)  # 把任务写到机器人字典中
#         bot_list[min_distance_bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
#         return 1  # 做出抢占


def task_schedule_in_preempting_priority(buy_table_type_list, preemption_distance_kp):
    """
    抢占优先级任务规划，每次执行最多发生一个抢占任务。
    :param buy_table_type_list: 抢占优先级购买工作台类型列表
    :param preemption_distance_kp: 抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
    :return: 0：不做抢占，1：做出抢占
    """
    global bot_list, table_list
    min_distance_sum = 200
    min_distance_table_id_1 = -1
    min_distance_table_id_2 = -1
    min_distance_object_type = 0
    min_distance_bot_id = -1
    # 先看有没有可以被抢占的机器人
    for bot in bot_list:
        # 如果有可以被抢占123的机器人
        if bot['work_status'] == 1 and bot['object'] == 0 and table_list[bot['target_table_id'][0]]['type'] in [1, 2,
                                                                                                                3]:
            preemption_buy_table_type_list = [1, 2, 3]
            break
    # 没有可以被抢占123的机器人
    else:
        # 如果抢占任务是7
        if buy_table_type_list == [7]:
            # 再找
            for bot in bot_list:
                # 如果有可以被抢占456的机器人也行
                if bot['work_status'] == 1 and bot['object'] == 0 and table_list[bot['target_table_id'][0]]['type'] in [
                    4, 5, 6]:
                    preemption_buy_table_type_list = [4, 5, 6]
                    break
            # 456也没有
            else:
                return 0
        # 不是7
        else:
            return 0
    # 遍历该抢占优先级的各种工作台
    for table_type_id_1 in buy_table_type_list:
        # 如果存在这种工作台
        if table_type_id_1 in table_type:
            # 遍历这种每个工作台
            for table_id_1 in table_index[f'type_{table_type_id_1}']:
                # 如果有产品未被分配取走任务
                if table_list[table_id_1]['projected_production'] == 1:
                    # 预检查是否需要通过优先级分配该购买任务
                    # 如果只由原地购买承担
                    if task_schedule_by_inplace_buy_precheck(table_id_1):
                        continue  # 不去这个工作台购买
                    # 如果需要由分析是否需要抢占优先级承担
                    else:
                        # 遍历机器人
                        for bot_id in range(bot_num):
                            # 抢占任务
                            if bot_list[bot_id]['work_status'] == 1 and bot_list[bot_id]['object'] == 0 and \
                                    table_list[bot_list[bot_id]['target_table_id'][0]][
                                        'type'] in preemption_buy_table_type_list:
                                remaining_distance = calculate_distance(bot_id, bot_list[bot_id]['target_table_id'][
                                    0])  # 计算该机器人与原本的购买工作台之间剩余的距离
                                preemption_distance = calculate_distance(bot_id, table_id_1)  # 计算该机器人与抢占任务购买工作台之间的距离
                                # 如果剩余距离较长
                                if remaining_distance >= preemption_distance_kp * preemption_distance:
                                    # 可以抢占
                                    # 寻找离本次选择的购买产品的工作台最近的可以出售该物品的工作台（1、2、3不到9卖）
                                    distance_1 = preemption_distance
                                    object_type = table_list[table_id_1]['type']  # 物品种类
                                    if object_type == 1:
                                        sell_table_type_list = [4, 5]
                                    elif object_type == 2:
                                        sell_table_type_list = [4, 6]
                                    elif object_type == 3:
                                        sell_table_type_list = [5, 6]
                                    elif object_type in [4, 5, 6]:
                                        sell_table_type_list = [7, 9]
                                    elif object_type == 7:
                                        sell_table_type_list = [8, 9]
                                    else:
                                        sell_table_type_list = []
                                    # 遍历可以出售该物品的各种工作台
                                    for table_type_id_2 in sell_table_type_list:
                                        # 如果存在这种工作台
                                        if table_type_id_2 in table_type:
                                            # 如果是8、9，不用判断原材料格
                                            if table_type_id_2 in [8, 9]:
                                                # 遍历这种每个工作台
                                                for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                                    distance_2 = calculate_distance_between_table(table_id_1,
                                                                                                  table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                                    distance_sum = distance_1 + distance_2  # 计算两段距离和
                                                    # 如果当前距离和小于存的最小距离和
                                                    if distance_sum < min_distance_sum:
                                                        min_distance_sum = distance_sum  # 更新最小距离和
                                                        min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                                        min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                                        min_distance_bot_id = bot_id  # 更新最小距离和下的机器人ID
                                                        min_distance_object_type = object_type  # 更新最小距离和下的物品类型
                                            # 如果不是8、9，要判断计划原材料格状态
                                            else:
                                                # 遍历这种每个工作台
                                                for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                                    # 如果缺少这个材料
                                                    if object_type in table_list[table_id_2][
                                                        'projected_missing_material']:
                                                        distance_2 = calculate_distance_between_table(table_id_1,
                                                                                                      table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                                        distance_sum = distance_1 + distance_2  # 计算两段距离和
                                                        # 如果当前距离和小于存的最小距离和
                                                        if distance_sum < min_distance_sum:
                                                            min_distance_sum = distance_sum  # 更新最小距离和
                                                            min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                                            min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                                            min_distance_bot_id = bot_id  # 更新最小距离和下的机器人ID
                                                            min_distance_object_type = object_type  # 更新最小距离和下的物品类型
    # 如果找不到需要购买的产品或找到了要购买的产品但是没地方卖
    if min_distance_table_id_1 == -1 or min_distance_table_id_2 == -1:
        return 0  # 不做抢占
    # 发生抢占
    else:
        # 复原原计划购买工作台的计划产品格状态
        table_list[bot_list[min_distance_bot_id]['target_table_id'][0]]['projected_production'] = 1
        # 复原原计划出售工作台的计划原材料格状态
        table_list[bot_list[min_distance_bot_id]['target_table_id'][1]]['projected_missing_material'].append(
            table_list[bot_list[min_distance_bot_id]['target_table_id'][0]]['type'])
        # 新任务
        table_list[min_distance_table_id_1]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
        # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
        if table_list[min_distance_table_id_2]['type'] not in [8, 9]:
            table_list[min_distance_table_id_2]['projected_missing_material'].remove(
                min_distance_object_type)  # 更新工作台的考虑计划的原料格状态
        bot_list[min_distance_bot_id]['target_table_id'] = (
            min_distance_table_id_1, min_distance_table_id_2)  # 把任务写到机器人字典中
        bot_list[min_distance_bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
        return 1  # 做出抢占


# 4/5/6动态优先级调度类，保证4/5/6数量平衡
class DynamicPrioritySchedule(object):
    def __init__(self):
        # 三类工作台合成数量计数器
        self.counter_4 = 0
        self.counter_5 = 0
        self.counter_6 = 0

    def counter_update(self):
        """
        更新动态优先级调度的产品计数器。
        :return: none
        """
        global table_list
        # 遍历所有工作台
        for table in table_list:
            # 如果是4、5、6类
            if table['type'] in [4, 5, 6]:
                # 如果开始了一次生产
                if table['material'] == 0 and table['pre_material'] != 0:
                    # 计数一次
                    if table['type'] == 4:
                        self.counter_4 += 1
                    elif table['type'] == 5:
                        self.counter_5 += 1
                    elif table['type'] == 6:
                        self.counter_6 += 1

    def schedule(self):
        """
        根据计数器的值返回当前比较缺的产品。
        :return: 4/5/6中较少的产品类别
        """
        # 如果数量均相等
        if self.counter_4 == self.counter_5 == self.counter_6:
            return 0  # 不需要动态优先级调度
        # 数量不相等
        else:
            counter_list = [self.counter_4, self.counter_5, self.counter_6]
            missing_production = numpy.argmin(counter_list)
            return missing_production + 4  # 需要动态优先级调度，返回需要合成的物品类别


def task_schedule_in_dynamic_priority(bot_id: int):
    """
    动态优先级任务规划，主要用于平衡4/5/6类产品的数量。
    :param bot_id: 机器人ID
    :return: 0：不做规划，1：做出规划
    """
    global bot_list, table_list, table_type, table_index, dynamic_priority
    min_distance_sum = 200
    min_distance_table_id_1 = -1
    min_distance_table_id_2 = -1
    min_distance_object_type = 0
    # 查询缺少的物品类型
    missing_production = dynamic_priority.schedule()
    # 根据缺少的开始动态优先级规划
    if missing_production == 4:
        buy_table_type_list = [1, 2]
    elif missing_production == 5:
        buy_table_type_list = [1, 3]
    elif missing_production == 6:
        buy_table_type_list = [2, 3]
    else:
        buy_table_type_list = []
    table_type_id_2 = missing_production
    # 遍历购买工作台
    for table_type_id_1 in buy_table_type_list:
        # 如果存在这种工作台
        if table_type_id_1 in table_type:
            # 遍历这种每个工作台
            for table_id_1 in table_index[f'type_{table_type_id_1}']:
                # 如果有产品未被分配取走任务
                if table_list[table_id_1]['projected_production'] == 1:
                    distance_1 = calculate_distance(bot_id, table_id_1)  # 计算机器人与该工作台的距离
                    # 寻找离本次选择的购买产品的工作台最近的可以出售该物品的工作台
                    # 如果存在这种工作台（4/5/6）
                    object_type = table_list[table_id_1]['type']
                    if table_type_id_2 in table_type:
                        # 遍历这种每个工作台
                        for table_id_2 in table_index[f'type_{table_type_id_2}']:
                            # 如果缺少这个材料
                            if object_type in table_list[table_id_2]['projected_missing_material']:
                                distance_2 = calculate_distance_between_table(table_id_1,
                                                                              table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                distance_sum = distance_1 + distance_2  # 计算两段距离和
                                # 如果当前距离和小于存的最小距离和
                                if distance_sum < min_distance_sum:
                                    min_distance_sum = distance_sum  # 更新最小距离和
                                    min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                    min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                    min_distance_object_type = object_type  # 更新最小距离和购买物品类型
    # 如果找不到需要购买的产品或找到了要购买的产品但是没地方卖
    if min_distance_table_id_1 == -1 or min_distance_table_id_2 == -1:
        return 0  # 不做规划
    else:
        table_list[min_distance_table_id_1]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
        table_list[min_distance_table_id_2]['projected_missing_material'].remove(
            min_distance_object_type)  # 更新工作台的考虑计划的原料格状态
        bot_list[bot_id]['target_table_id'] = (min_distance_table_id_1, min_distance_table_id_2)  # 把任务写到机器人字典中
        bot_list[bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
        return 1  # 做出规划


def task_schedule():
    """
    机器人总体任务规划。
    :return: none
    """
    global bot_list
    # 抢占优先级
    task_schedule_in_preempting_priority([7], 0.6)
    task_schedule_in_preempting_priority([4, 5, 6], 0.6)
    # 更新动态优先级调度计数器
    dynamic_priority.counter_update()
    # 遍历机器人
    for bot_id in range(bot_num):
        # 如果机器人空闲则规划任务
        if bot_list[bot_id]['work_status'] == 0:
            # 先判断原地是否能够买
            # 可能存在抢占的情况
            if task_schedule_by_inplace_buy(bot_id, 10):
                continue
            # 动态优先级调度
            if task_schedule_in_dynamic_priority(bot_id):
                continue
            # 按从高到低的优先级遍历工作台是否有产品
            # 工作台产品优先级为(7)(654)(321)
            # 第一优先级(7)
            if task_schedule_in_the_same_priority(bot_id, [7]):
                continue
            # 第二优先级(456)
            if task_schedule_in_the_same_priority(bot_id, [4, 5, 6]):
                continue
            # 第三优先级(123)
            if task_schedule_in_the_same_priority(bot_id, [1, 2, 3]):
                continue


# endregion


# region 针对性算法，在使用针对性算法时不适用通用的task_schedule()等
# 藏在获取数据中，在get_init_data()内被调用
def get_map_num():
    """
    获取地图编号以便于针对性操作。
    :return: none
    """
    global table_list, map_num
    if table_list[0]['type'] == 1:
        map_num = 1
    elif table_list[0]['type'] == 6:
        map_num = 2
    elif table_list[0]['type'] == 3:
        map_num = 3
    elif table_list[0]['type'] == 7:
        map_num = 4
    else:
        map_num = 0


# 藏在买卖控制中，在control()中被调用
def stop_work():
    """
    到达地图最高分时停止一切工作。
    :return: none
    """
    if (map_num == 1 and money >= 819965) or (map_num == 2 and money >= 700014) or (
            map_num == 3 and money >= 571665) or (map_num == 4 and money >= 659498):
        command['forward'] = [0, 0, 0, 0]
        command['rotate'] = [0, 0, 0, 0]
        command['buy'] = []
        command['sell'] = []
        command['destroy'] = []
        send_data()
        while True:
            try:
                update_data()
                send_data()
            except EOFError:  # 读到9000帧后下一帧是EOF,此时停止程序
                exit()


# ban掉工作台
def ban_table():
    """
    禁用部分工作台
    :return: none
    """
    global table_list
    if map_num == 1:
        ban_list = [10, 12, 17, 15, 22, 21, 2, 5, 7, 4, 9, 13, 27, 20, 26, 28, 29, 35, 36, 37, 25, 31, 39, 32, 8, 1, 16,
                    24, 34, 33, 6]
        for ban_id in ban_list:
            table_list[ban_id]['type'] = 0
    elif map_num == 2:
        ban_list = [2, 9, 15, 22]
        for ban_id in ban_list:
            table_list[ban_id]['type'] = 0
    elif map_num == 3:
        ban_list = [8, 9, 10, 30, 1, 3, 5, 11, 12, 13, 14, 32, 33, 34, 29, 44, 47, 48]
        for ban_id in ban_list:
            table_list[ban_id]['type'] = 0
    elif map_num == 4:
        pass


def task_schedule_in_the_same_priority_map_1_special(bot_id: int, buy_table_type_list):
    """
    从同等优先级的工作台购买产品并出售的任务规划，针对地图1的特别版。
    :param bot_id: 机器人ID
    :param buy_table_type_list: 该优先级的工作台类型
    :return: 0：不做规划，1：做出规划
    """
    global bot_list, table_list, table_type, table_index
    min_distance_sum = 200
    min_distance_table_id_1 = -1
    min_distance_table_id_2 = -1
    min_distance_object_type = 0
    # 遍历该优先级的各种工作台
    for table_type_id_1 in buy_table_type_list:
        # 如果存在这种工作台
        if table_type_id_1 in table_type:
            # 遍历这种每个工作台
            for table_id_1 in table_index[f'type_{table_type_id_1}']:
                # 如果有产品未被分配取走任务
                if table_list[table_id_1]['projected_production'] == 1:
                    # 预检查是否需要通过优先级分配该购买任务
                    # 如果只由原地购买承担
                    if task_schedule_by_inplace_buy_precheck(table_id_1):
                        continue  # 不去这个工作台购买
                    # 如果需要由优先级承担
                    else:
                        distance_1 = calculate_distance(bot_id, table_id_1)  # 计算机器人与该工作台的距离
                        # 寻找离本次选择的购买产品的工作台最近的可以出售该物品的工作台（1、2、3不到9卖）
                        # 地图1特别版，4/5不卖到9，6优先卖到7其次9
                        object_type = table_list[table_id_1]['type']  # 物品种类
                        if object_type == 1:
                            sell_table_type_list = [4, 5]
                        elif object_type == 2:
                            sell_table_type_list = [4, 6]
                        elif object_type == 3:
                            sell_table_type_list = [5, 6]
                        elif object_type in [4, 5, 6]:
                            sell_table_type_list = [7]
                        elif object_type == 7:
                            sell_table_type_list = [8, 9]
                        else:
                            sell_table_type_list = []
                        # 遍历可以出售该物品的各种工作台
                        for table_type_id_2 in sell_table_type_list:
                            # 如果存在这种工作台
                            if table_type_id_2 in table_type:
                                # 如果是8、9，不用判断原材料格
                                if table_type_id_2 in [8, 9]:
                                    # 遍历这种每个工作台
                                    for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                        distance_2 = calculate_distance_between_table(table_id_1,
                                                                                      table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                        distance_sum = distance_1 + distance_2  # 计算两段距离和
                                        # 如果当前距离和小于存的最小距离和
                                        if distance_sum < min_distance_sum:
                                            min_distance_sum = distance_sum  # 更新最小距离和
                                            min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                            min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                            min_distance_object_type = object_type  # 更新最小距离和下的物品类型
                                # 如果不是8、9，要判断计划原材料格状态
                                else:
                                    # 遍历这种每个工作台
                                    for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                        # 如果缺少这个材料
                                        if object_type in table_list[table_id_2]['projected_missing_material']:
                                            distance_2 = calculate_distance_between_table(table_id_1,
                                                                                          table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                            distance_sum = distance_1 + distance_2  # 计算两段距离和
                                            # 如果当前距离和小于存的最小距离和
                                            if distance_sum < min_distance_sum:
                                                min_distance_sum = distance_sum  # 更新最小距离和
                                                min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                                min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                                min_distance_object_type = object_type  # 更新最小距离和下的物品类型
    # 如果需要购买的是6而没7卖
    if 6 in buy_table_type_list and (min_distance_table_id_1 == -1 or min_distance_table_id_2 == -1):
        table_type_id_1 = 6
        # 如果存在这种工作台
        if table_type_id_1 in table_type:
            # 遍历这种每个工作台
            for table_id_1 in table_index[f'type_{table_type_id_1}']:
                # 如果有产品未被分配取走任务
                if table_list[table_id_1]['projected_production'] == 1:
                    # 预检查是否需要通过优先级分配该购买任务
                    # 如果只由原地购买承担
                    if task_schedule_by_inplace_buy_precheck(table_id_1):
                        continue  # 不去这个工作台购买
                    # 如果需要由优先级承担
                    else:
                        distance_1 = calculate_distance(bot_id, table_id_1)  # 计算机器人与该工作台的距离
                        # 寻找离本次选择的购买产品的工作台最近的可以出售该物品的工作台（1、2、3不到9卖）
                        # 地图1特别版，4/5不卖到9，6优先卖到7其次9
                        object_type = table_list[table_id_1]['type']  # 物品种类
                        sell_table_type_list = [9]
                        # 遍历可以出售该物品的各种工作台，卖到9
                        for table_type_id_2 in sell_table_type_list:
                            # 如果存在这种工作台
                            if table_type_id_2 in table_type:
                                # 遍历这种每个工作台
                                for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                    distance_2 = calculate_distance_between_table(table_id_1,
                                                                                  table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                    distance_sum = distance_1 + distance_2  # 计算两段距离和
                                    # 如果当前距离和小于存的最小距离和
                                    if distance_sum < min_distance_sum:
                                        min_distance_sum = distance_sum  # 更新最小距离和
                                        min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                        min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                        min_distance_object_type = object_type  # 更新最小距离和下的物品类型
    # 如果找不到需要购买的产品或找到了要购买的产品但是没地方卖
    if min_distance_table_id_1 == -1 or min_distance_table_id_2 == -1:
        return 0  # 不做规划
    else:
        table_list[min_distance_table_id_1]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
        # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
        if table_list[min_distance_table_id_2]['type'] not in [8, 9]:
            table_list[min_distance_table_id_2]['projected_missing_material'].remove(
                min_distance_object_type)  # 更新工作台的考虑计划的原料格状态
        bot_list[bot_id]['target_table_id'] = (min_distance_table_id_1, min_distance_table_id_2)  # 把任务写到机器人字典中
        bot_list[bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
        return 1  # 做出规划


def task_schedule_by_inplace_buy_map_1_special(bot_id: int, wait_threshold):
    """
    机器人出售后如果工作台有产品的话直接原地购买，若存在抢占情况则把被抢占该购买任务的机器人的工作状态改为空闲同时停止运动，针对地图1的特别版。
    :param bot_id: 机器人ID
    :param wait_threshold: 最长等待时间
    :return: 0：不做规划，1：做出规划
    """
    global bot_list, table_list
    # 如果机器人在工作台处时空闲
    if bot_list[bot_id]['table_id'] != -1:
        inplace_buy_table_id = bot_list[bot_id]['table_id']  # 所在工作台ID
        # 如果是4/5/6/7类工作台（因为不可能在1/2/3类工作台空闲，在8/9则是不可能有产品）
        if table_list[inplace_buy_table_id]['type'] in [4, 5, 6, 7]:
            # 如果这个工作台有产品（非计划）
            if table_list[inplace_buy_table_id]['production'] == 1:
                # 即使有预检查也可能存在购买抢占的情况，遍历其他机器人看看有没有要买这个工作台产品的任务
                for bot_id_2 in range(bot_num):
                    # 如果不是当前机器人
                    if bot_id_2 != bot_id:
                        # 如果其购买任务被抢占
                        if bot_list[bot_id_2]['target_table_id'][0] == inplace_buy_table_id and bot_list[bot_id_2][
                            'work_status'] == 1:
                            # 停止运动
                            command['forward'][bot_id_2] = 0
                            command['rotate'][bot_id_2] = 0
                            # 修改运动状态为空闲
                            bot_list[bot_id_2]['work_status'] = 0
                            # 当前机器人直接购买并设置出售目标工作台为被抢占机器人的出售目标工作台
                            bot_list[bot_id]['target_table_id'] = (
                                inplace_buy_table_id, bot_list[bot_id_2]['target_table_id'][1])  # 把任务写到当前机器人种
                            bot_list[bot_id]['work_status'] = 1  # 当前机器人运动状态改为前往第一个目标工作台
                            return 1  # 做出规划
                # 没有机器人的任务被抢占
                # 寻找最近的可以出售该物品的工作台，4/5不卖到9，6优先7再9
                object_type = table_list[inplace_buy_table_id]['type']  # 物品种类
                if object_type in [4, 5, 6]:
                    sell_table_type_list = [7]
                elif object_type == 7:
                    sell_table_type_list = [8, 9]
                else:
                    sell_table_type_list = []
                min_distance = 100
                min_distance_table_id = -1
                # 遍历可以出售该物品的各种工作台
                for table_type_id in sell_table_type_list:
                    # 如果存在这种工作台
                    if table_type_id in table_type:
                        # 如果是8、9，不用判断原材料格
                        if table_type_id in [8, 9]:
                            # 遍历这种每个工作台
                            for table_id in table_index[f'type_{table_type_id}']:
                                distance = calculate_distance_between_table(inplace_buy_table_id,
                                                                            table_id)  # 计算购买工作台与出售工作台之间的距离
                                # 如果比目前存的最小距离小
                                if distance < min_distance:
                                    min_distance = distance  # 更新最小距离
                                    min_distance_table_id = table_id  # 更新最近工作台ID
                        # 如果不是8、9，要判断计划原材料格状态
                        else:
                            # 遍历这种每个工作台
                            for table_id in table_index[f'type_{table_type_id}']:
                                # 如果缺少这个材料
                                if object_type in table_list[table_id]['projected_missing_material']:
                                    distance = calculate_distance_between_table(inplace_buy_table_id,
                                                                                table_id)  # 计算购买工作台与出售工作台之间的距离
                                    # 如果比目前存的最小距离小
                                    if distance < min_distance:
                                        min_distance = distance  # 更新最小距离
                                        min_distance_table_id = table_id  # 更新最近工作台ID
                # 如果需要买的是6且没有7可以去卖
                if object_type == 6 and min_distance_table_id == -1:
                    sell_table_type_list = [9]
                    min_distance = 100
                    min_distance_table_id = -1
                    # 遍历可以出售该物品的各种工作台
                    for table_type_id in sell_table_type_list:
                        # 如果存在这种工作台
                        if table_type_id in table_type:
                            # 遍历这种每个工作台
                            for table_id in table_index[f'type_{table_type_id}']:
                                distance = calculate_distance_between_table(inplace_buy_table_id,
                                                                            table_id)  # 计算购买工作台与出售工作台之间的距离
                                # 如果比目前存的最小距离小
                                if distance < min_distance:
                                    min_distance = distance  # 更新最小距离
                                    min_distance_table_id = table_id  # 更新最近工作台ID
                # 没地方卖
                if min_distance_table_id == -1:
                    return 0  # 不做规划
                else:
                    table_list[inplace_buy_table_id]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
                    # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
                    if table_list[min_distance_table_id]['type'] not in [8, 9]:
                        table_list[min_distance_table_id]['projected_missing_material'].remove(
                            object_type)  # 更新工作台的考虑计划的原料格状态
                    # 修改机器人任务
                    bot_list[bot_id]['target_table_id'] = (
                        inplace_buy_table_id, min_distance_table_id)  # 把任务写到机器人字典中
                    bot_list[bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
                    return 1  # 做出规划
            # 如果这个工作台没产品
            else:
                # 如果剩余时间在等待阈值内
                if -1 < table_list[inplace_buy_table_id]['remaining_time'] < wait_threshold:
                    return 1  # 等待，假装做出规划
                return 0  # 不做规划


def task_schedule_in_preempting_priority_map_1_special(buy_table_type_list, preemption_distance_kp):
    """
    抢占优先级任务规划，每次执行最多发生一个抢占任务，针对地图1的特别版。
    :param buy_table_type_list: 抢占优先级购买工作台类型列表
    :param preemption_distance_kp: 抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
    :return: 0：不做抢占，1：做出抢占
    """
    global bot_list, table_list
    min_distance_sum = 200
    min_distance_table_id_1 = -1
    min_distance_table_id_2 = -1
    min_distance_object_type = 0
    min_distance_bot_id = -1
    # 先看有没有可以被抢占的机器人
    for bot in bot_list:
        # 如果有可以被抢占123的机器人
        if bot['work_status'] == 1 and bot['object'] == 0 and table_list[bot['target_table_id'][0]]['type'] in [1, 2,
                                                                                                                3]:
            preemption_buy_table_type_list = [1, 2, 3]
            break
    # 没有可以被抢占123的机器人
    else:
        # 如果抢占任务是7
        if buy_table_type_list == [7]:
            # 再找
            for bot in bot_list:
                # 如果有可以被抢占456的机器人也行
                if bot['work_status'] == 1 and bot['object'] == 0 and table_list[bot['target_table_id'][0]]['type'] in [
                    4, 5, 6]:
                    preemption_buy_table_type_list = [4, 5, 6]
                    break
            # 456也没有
            else:
                return 0
        # 不是7
        else:
            return 0
    # 遍历该抢占优先级的各种工作台
    for table_type_id_1 in buy_table_type_list:
        # 如果存在这种工作台
        if table_type_id_1 in table_type:
            # 遍历这种每个工作台
            for table_id_1 in table_index[f'type_{table_type_id_1}']:
                # 如果有产品未被分配取走任务
                if table_list[table_id_1]['projected_production'] == 1:
                    # 预检查是否需要通过优先级分配该购买任务
                    # 如果只由原地购买承担
                    if task_schedule_by_inplace_buy_precheck(table_id_1):
                        continue  # 不去这个工作台购买
                    # 如果需要由分析是否需要抢占优先级承担
                    else:
                        # 遍历机器人
                        for bot_id in range(bot_num):
                            # 抢占任务
                            if bot_list[bot_id]['work_status'] == 1 and bot_list[bot_id]['object'] == 0 and \
                                    table_list[bot_list[bot_id]['target_table_id'][0]][
                                        'type'] in preemption_buy_table_type_list:
                                remaining_distance = calculate_distance(bot_id, bot_list[bot_id]['target_table_id'][
                                    0])  # 计算该机器人与原本的购买工作台之间剩余的距离
                                preemption_distance = calculate_distance(bot_id, table_id_1)  # 计算该机器人与抢占任务购买工作台之间的距离
                                # 如果剩余距离较长
                                if remaining_distance >= preemption_distance_kp * preemption_distance:
                                    # 可以抢占
                                    # 寻找离本次选择的购买产品的工作台最近的可以出售该物品的工作台（1、2、3、4、5不到9卖，6优先7其次9）
                                    distance_1 = preemption_distance
                                    object_type = table_list[table_id_1]['type']  # 物品种类
                                    if object_type == 1:
                                        sell_table_type_list = [4, 5]
                                    elif object_type == 2:
                                        sell_table_type_list = [4, 6]
                                    elif object_type == 3:
                                        sell_table_type_list = [5, 6]
                                    elif object_type in [4, 5, 6]:
                                        sell_table_type_list = [7]
                                    elif object_type == 7:
                                        sell_table_type_list = [8, 9]
                                    else:
                                        sell_table_type_list = []
                                    # 遍历可以出售该物品的各种工作台
                                    for table_type_id_2 in sell_table_type_list:
                                        # 如果存在这种工作台
                                        if table_type_id_2 in table_type:
                                            # 如果是8、9，不用判断原材料格
                                            if table_type_id_2 in [8, 9]:
                                                # 遍历这种每个工作台
                                                for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                                    distance_2 = calculate_distance_between_table(table_id_1,
                                                                                                  table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                                    distance_sum = distance_1 + distance_2  # 计算两段距离和
                                                    # 如果当前距离和小于存的最小距离和
                                                    if distance_sum < min_distance_sum:
                                                        min_distance_sum = distance_sum  # 更新最小距离和
                                                        min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                                        min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                                        min_distance_bot_id = bot_id  # 更新最小距离和下的机器人ID
                                                        min_distance_object_type = object_type  # 更新最小距离和下的物品类型
                                            # 如果不是8、9，要判断计划原材料格状态
                                            else:
                                                # 遍历这种每个工作台
                                                for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                                    # 如果缺少这个材料
                                                    if object_type in table_list[table_id_2][
                                                        'projected_missing_material']:
                                                        distance_2 = calculate_distance_between_table(table_id_1,
                                                                                                      table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                                        distance_sum = distance_1 + distance_2  # 计算两段距离和
                                                        # 如果当前距离和小于存的最小距离和
                                                        if distance_sum < min_distance_sum:
                                                            min_distance_sum = distance_sum  # 更新最小距离和
                                                            min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                                            min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                                            min_distance_bot_id = bot_id  # 更新最小距离和下的机器人ID
                                                            min_distance_object_type = object_type  # 更新最小距离和下的物品类型
    # 如果要购买的是6且没7可以去卖
    if 6 in buy_table_type_list and (min_distance_table_id_1 == -1 or min_distance_table_id_2 == -1):
        table_type_id_1 = 6
        # 如果存在这种工作台
        if table_type_id_1 in table_type:
            # 遍历这种每个工作台
            for table_id_1 in table_index[f'type_{table_type_id_1}']:
                # 如果有产品未被分配取走任务
                if table_list[table_id_1]['projected_production'] == 1:
                    # 预检查是否需要通过优先级分配该购买任务
                    # 如果只由原地购买承担
                    if task_schedule_by_inplace_buy_precheck(table_id_1):
                        continue  # 不去这个工作台购买
                    # 如果需要由分析是否需要抢占优先级承担
                    else:
                        # 遍历机器人
                        for bot_id in range(bot_num):
                            # 抢占任务
                            if bot_list[bot_id]['work_status'] == 1 and bot_list[bot_id]['object'] == 0 and \
                                    table_list[bot_list[bot_id]['target_table_id'][0]][
                                        'type'] in preemption_buy_table_type_list:
                                remaining_distance = calculate_distance(bot_id, bot_list[bot_id]['target_table_id'][
                                    0])  # 计算该机器人与原本的购买工作台之间剩余的距离
                                preemption_distance = calculate_distance(bot_id, table_id_1)  # 计算该机器人与抢占任务购买工作台之间的距离
                                # 如果剩余距离较长
                                if remaining_distance >= preemption_distance_kp * preemption_distance:
                                    # 可以抢占
                                    # 寻找离本次选择的购买产品的工作台最近的可以出售该物品的工作台（1、2、3、4、5不到9卖，6优先7其次9）
                                    distance_1 = preemption_distance
                                    object_type = table_list[table_id_1]['type']  # 物品种类
                                    sell_table_type_list = [9]
                                    # 遍历可以出售该物品的各种工作台
                                    for table_type_id_2 in sell_table_type_list:
                                        # 如果存在这种工作台
                                        if table_type_id_2 in table_type:
                                            # 遍历这种每个工作台
                                            for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                                distance_2 = calculate_distance_between_table(table_id_1,
                                                                                              table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                                distance_sum = distance_1 + distance_2  # 计算两段距离和
                                                # 如果当前距离和小于存的最小距离和
                                                if distance_sum < min_distance_sum:
                                                    min_distance_sum = distance_sum  # 更新最小距离和
                                                    min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                                    min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                                    min_distance_bot_id = bot_id  # 更新最小距离和下的机器人ID
                                                    min_distance_object_type = object_type  # 更新最小距离和下的物品类型
    # 如果找不到需要购买的产品或找到了要购买的产品但是没地方卖
    if min_distance_table_id_1 == -1 or min_distance_table_id_2 == -1:
        return 0  # 不做抢占
    # 发生抢占
    else:
        # 复原原计划购买工作台的计划产品格状态
        table_list[bot_list[min_distance_bot_id]['target_table_id'][0]]['projected_production'] = 1
        # 复原原计划出售工作台的计划原材料格状态
        table_list[bot_list[min_distance_bot_id]['target_table_id'][1]]['projected_missing_material'].append(
            table_list[bot_list[min_distance_bot_id]['target_table_id'][0]]['type'])
        # 新任务
        table_list[min_distance_table_id_1]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
        # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
        if table_list[min_distance_table_id_2]['type'] not in [8, 9]:
            table_list[min_distance_table_id_2]['projected_missing_material'].remove(
                min_distance_object_type)  # 更新工作台的考虑计划的原料格状态
        bot_list[min_distance_bot_id]['target_table_id'] = (
            min_distance_table_id_1, min_distance_table_id_2)  # 把任务写到机器人字典中
        bot_list[min_distance_bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
        return 1  # 做出抢占


def task_schedule_in_the_same_priority_map_3_special(bot_id: int, bot_work_area, buy_table_type_list):
    """
    从同等优先级的工作台购买产品并出售的任务规划，针对地图3的特别版。
    :param bot_id: 机器人ID
    :param bot_work_area: 机器人工作区域字典
    :param buy_table_type_list: 该优先级的工作台类型
    :return: 0：不做规划，1：做出规划
    """
    global bot_list, table_list, table_type, table_index
    min_distance_sum = 200
    min_distance_table_id_1 = -1
    min_distance_table_id_2 = -1
    min_distance_object_type = 0
    # 遍历该优先级的各种工作台
    for table_type_id_1 in buy_table_type_list:
        # 如果存在这种工作台
        if table_type_id_1 in table_type:
            # 遍历这种每个工作台
            for table_id_1 in table_index[f'type_{table_type_id_1}']:
                # 如果该工作台在该机器人的工作区域
                if table_id_1 in bot_work_area[f'bot_{bot_id}']:
                    # 如果有产品未被分配取走任务
                    if table_list[table_id_1]['projected_production'] == 1:
                        # 预检查是否需要通过优先级分配该购买任务
                        # 如果只由原地购买承担
                        if task_schedule_by_inplace_buy_precheck(table_id_1):
                            continue  # 不去这个工作台购买
                        # 如果需要由优先级承担
                        else:
                            distance_1 = calculate_distance(bot_id, table_id_1)  # 计算机器人与该工作台的距离
                            # 寻找离本次选择的购买产品的工作台最近的可以出售该物品的工作台（1、2、3不到9卖）
                            object_type = table_list[table_id_1]['type']  # 物品种类
                            if object_type == 1:
                                sell_table_type_list = [4, 5]
                            elif object_type == 2:
                                sell_table_type_list = [4, 6]
                            elif object_type == 3:
                                sell_table_type_list = [5, 6]
                            elif object_type in [4, 5, 6]:
                                sell_table_type_list = [7, 9]
                            elif object_type == 7:
                                sell_table_type_list = [8, 9]
                            else:
                                sell_table_type_list = []
                            # 遍历可以出售该物品的各种工作台
                            for table_type_id_2 in sell_table_type_list:
                                # 如果存在这种工作台
                                if table_type_id_2 in table_type:
                                    # 如果是8、9，不用判断原材料格
                                    if table_type_id_2 in [8, 9]:
                                        # 遍历这种每个工作台
                                        for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                            # 如果工作台在机器人的工作区域
                                            if table_id_2 in bot_work_area[f'bot_{bot_id}']:
                                                distance_2 = calculate_distance_between_table(table_id_1,
                                                                                              table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                                distance_sum = distance_1 + distance_2  # 计算两段距离和
                                                # 如果当前距离和小于存的最小距离和
                                                if distance_sum < min_distance_sum:
                                                    min_distance_sum = distance_sum  # 更新最小距离和
                                                    min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                                    min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                                    min_distance_object_type = object_type  # 更新最小距离和下的物品类型
                                    # 如果不是8、9，要判断计划原材料格状态
                                    else:
                                        # 遍历这种每个工作台
                                        for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                            # 如果工作台在机器人的工作区域
                                            if table_id_2 in bot_work_area[f'bot_{bot_id}']:
                                                # 如果缺少这个材料
                                                if object_type in table_list[table_id_2]['projected_missing_material']:
                                                    distance_2 = calculate_distance_between_table(table_id_1,
                                                                                                  table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                                    distance_sum = distance_1 + distance_2  # 计算两段距离和
                                                    # 如果当前距离和小于存的最小距离和
                                                    if distance_sum < min_distance_sum:
                                                        min_distance_sum = distance_sum  # 更新最小距离和
                                                        min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                                        min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                                        min_distance_object_type = object_type  # 更新最小距离和下的物品类型
    # 如果找不到需要购买的产品或找到了要购买的产品但是没地方卖
    if min_distance_table_id_1 == -1 or min_distance_table_id_2 == -1:
        return 0  # 不做规划
    else:
        table_list[min_distance_table_id_1]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
        # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
        if table_list[min_distance_table_id_2]['type'] not in [8, 9]:
            table_list[min_distance_table_id_2]['projected_missing_material'].remove(
                min_distance_object_type)  # 更新工作台的考虑计划的原料格状态
        bot_list[bot_id]['target_table_id'] = (min_distance_table_id_1, min_distance_table_id_2)  # 把任务写到机器人字典中
        bot_list[bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
        return 1  # 做出规划


def task_schedule_by_inplace_buy_map_3_special(bot_id: int, bot_work_area, wait_threshold):
    """
    机器人出售后如果工作台有产品的话直接原地购买，若存在抢占情况则把被抢占该购买任务的机器人的工作状态改为空闲同时停止运动，针对地图3的特别版。
    :param bot_id: 机器人ID
    :param bot_work_area: 机器人工作区间字典
    :param wait_threshold: 最长等待时间
    :return: 0：不做规划，1：做出规划
    """
    global bot_list, table_list
    # 如果机器人在工作台处时空闲
    if bot_list[bot_id]['table_id'] != -1:
        inplace_buy_table_id = bot_list[bot_id]['table_id']  # 所在工作台ID
        # 如果是4/5/6/7类工作台（因为不可能在1/2/3类工作台空闲，在8/9则是不可能有产品）
        if table_list[inplace_buy_table_id]['type'] in [4, 5, 6, 7]:
            # 如果这个工作台有产品（非计划）
            if table_list[inplace_buy_table_id]['production'] == 1:
                # 即使有预检查也可能存在购买抢占的情况，遍历其他机器人看看有没有要买这个工作台产品的任务
                for bot_id_2 in range(bot_num):
                    # 如果不是当前机器人
                    if bot_id_2 != bot_id:
                        # 如果其购买任务被抢占（不需要特别处理，因为被抢占的肯定也是同一区域内的机器人）
                        if bot_list[bot_id_2]['target_table_id'][0] == inplace_buy_table_id and bot_list[bot_id_2][
                            'work_status'] == 1:
                            # 停止运动
                            command['forward'][bot_id_2] = 0
                            command['rotate'][bot_id_2] = 0
                            # 修改运动状态为空闲
                            bot_list[bot_id_2]['work_status'] = 0
                            # 当前机器人直接购买并设置出售目标工作台为被抢占机器人的出售目标工作台
                            bot_list[bot_id]['target_table_id'] = (
                                inplace_buy_table_id, bot_list[bot_id_2]['target_table_id'][1])  # 把任务写到当前机器人种
                            bot_list[bot_id]['work_status'] = 1  # 当前机器人运动状态改为前往第一个目标工作台
                            return 1  # 做出规划
                # 没有机器人的任务被抢占
                # 寻找最近的可以出售该物品的工作台
                object_type = table_list[inplace_buy_table_id]['type']  # 物品种类
                if object_type in [4, 5, 6]:
                    sell_table_type_list = [7, 9]
                elif object_type == 7:
                    sell_table_type_list = [8, 9]
                else:
                    sell_table_type_list = []
                min_distance = 100
                min_distance_table_id = -1
                # 遍历可以出售该物品的各种工作台
                for table_type_id in sell_table_type_list:
                    # 如果存在这种工作台
                    if table_type_id in table_type:
                        # 如果是8、9，不用判断原材料格
                        if table_type_id in [8, 9]:
                            # 遍历这种每个工作台
                            for table_id in table_index[f'type_{table_type_id}']:
                                # 如果在工作区间内
                                if table_id in bot_work_area[f'bot_{bot_id}']:
                                    distance = calculate_distance_between_table(inplace_buy_table_id,
                                                                                table_id)  # 计算购买工作台与出售工作台之间的距离
                                    # 如果比目前存的最小距离小
                                    if distance < min_distance:
                                        min_distance = distance  # 更新最小距离
                                        min_distance_table_id = table_id  # 更新最近工作台ID
                        # 如果不是8、9，要判断计划原材料格状态
                        else:
                            # 遍历这种每个工作台
                            for table_id in table_index[f'type_{table_type_id}']:
                                # 如果在工作区间内
                                if table_id in bot_work_area[f'bot_{bot_id}']:
                                    # 如果缺少这个材料
                                    if object_type in table_list[table_id]['projected_missing_material']:
                                        distance = calculate_distance_between_table(inplace_buy_table_id,
                                                                                    table_id)  # 计算购买工作台与出售工作台之间的距离
                                        # 如果比目前存的最小距离小
                                        if distance < min_distance:
                                            min_distance = distance  # 更新最小距离
                                            min_distance_table_id = table_id  # 更新最近工作台ID
                # 没地方卖
                if min_distance_table_id == -1:
                    return 0  # 不做规划
                else:
                    table_list[inplace_buy_table_id]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
                    # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
                    if table_list[min_distance_table_id]['type'] not in [8, 9]:
                        table_list[min_distance_table_id]['projected_missing_material'].remove(
                            object_type)  # 更新工作台的考虑计划的原料格状态
                    # 修改机器人任务
                    bot_list[bot_id]['target_table_id'] = (
                        inplace_buy_table_id, min_distance_table_id)  # 把任务写到机器人字典中
                    bot_list[bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
                    return 1  # 做出规划
            # 如果这个工作台没产品
            else:
                # 如果剩余时间在等待阈值内
                if -1 < table_list[inplace_buy_table_id]['remaining_time'] < wait_threshold:
                    return 1  # 等待，假装做出规划
                return 0  # 不做规划


def task_schedule_in_preempting_priority_map_3_special(buy_table_type_list, bot_work_area, preemption_distance_kp):
    """
    抢占优先级任务规划，每次执行最多发生一个抢占任务，针对地图3的特别版。
    :param buy_table_type_list: 抢占优先级购买工作台类型列表
    :param bot_work_area: 机器人工作区间字典
    :param preemption_distance_kp: 抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
    :return: 0：不做抢占，1：做出抢占
    """
    global bot_list, table_list
    min_distance_sum = 200
    min_distance_table_id_1 = -1
    min_distance_table_id_2 = -1
    min_distance_object_type = 0
    min_distance_bot_id = -1
    # 先看有没有可以被抢占的机器人
    for bot in bot_list:
        # 如果有可以被抢占123的机器人
        if bot['work_status'] == 1 and bot['object'] == 0 and table_list[bot['target_table_id'][0]]['type'] in [1, 2,
                                                                                                                3]:
            preemption_buy_table_type_list = [1, 2, 3]
            break
    # 没有可以被抢占123的机器人
    else:
        # 如果抢占任务是7
        if buy_table_type_list == [7]:
            # 再找
            for bot in bot_list:
                # 如果有可以被抢占456的机器人也行
                if bot['work_status'] == 1 and bot['object'] == 0 and table_list[bot['target_table_id'][0]]['type'] in [
                    4, 5, 6]:
                    preemption_buy_table_type_list = [4, 5, 6]
                    break
            # 456也没有
            else:
                return 0
        # 不是7
        else:
            return 0
    # 遍历该抢占优先级的各种工作台
    for table_type_id_1 in buy_table_type_list:
        # 如果存在这种工作台
        if table_type_id_1 in table_type:
            # 遍历这种每个工作台
            for table_id_1 in table_index[f'type_{table_type_id_1}']:
                # 如果有产品未被分配取走任务
                if table_list[table_id_1]['projected_production'] == 1:
                    # 预检查是否需要通过优先级分配该购买任务
                    # 如果只由原地购买承担
                    if task_schedule_by_inplace_buy_precheck(table_id_1):
                        continue  # 不去这个工作台购买
                    # 如果需要由分析是否需要抢占优先级承担
                    else:
                        # 遍历机器人
                        for bot_id in range(bot_num):
                            # 如果需要抢占的工作台在工作区间内
                            if table_id_1 in bot_work_area[f'bot_{bot_id}']:
                                # 抢占任务
                                if bot_list[bot_id]['work_status'] == 1 and bot_list[bot_id]['object'] == 0 and \
                                        table_list[bot_list[bot_id]['target_table_id'][0]][
                                            'type'] in preemption_buy_table_type_list:
                                    remaining_distance = calculate_distance(bot_id, bot_list[bot_id]['target_table_id'][
                                        0])  # 计算该机器人与原本的购买工作台之间剩余的距离
                                    preemption_distance = calculate_distance(bot_id,
                                                                             table_id_1)  # 计算该机器人与抢占任务购买工作台之间的距离
                                    # 如果剩余距离较长
                                    if remaining_distance >= preemption_distance_kp * preemption_distance:
                                        # 可以抢占
                                        # 寻找离本次选择的购买产品的工作台最近的可以出售该物品的工作台（1、2、3不到9卖）
                                        distance_1 = preemption_distance
                                        object_type = table_list[table_id_1]['type']  # 物品种类
                                        if object_type == 1:
                                            sell_table_type_list = [4, 5]
                                        elif object_type == 2:
                                            sell_table_type_list = [4, 6]
                                        elif object_type == 3:
                                            sell_table_type_list = [5, 6]
                                        elif object_type in [4, 5, 6]:
                                            sell_table_type_list = [7, 9]
                                        elif object_type == 7:
                                            sell_table_type_list = [8, 9]
                                        else:
                                            sell_table_type_list = []
                                        # 遍历可以出售该物品的各种工作台
                                        for table_type_id_2 in sell_table_type_list:
                                            # 如果存在这种工作台
                                            if table_type_id_2 in table_type:
                                                # 如果是8、9，不用判断原材料格
                                                if table_type_id_2 in [8, 9]:
                                                    # 遍历这种每个工作台
                                                    for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                                        # 如果在工作区域内
                                                        if table_id_2 in bot_work_area[f'bot_{bot_id}']:
                                                            distance_2 = calculate_distance_between_table(table_id_1,
                                                                                                          table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                                            distance_sum = distance_1 + distance_2  # 计算两段距离和
                                                            # 如果当前距离和小于存的最小距离和
                                                            if distance_sum < min_distance_sum:
                                                                min_distance_sum = distance_sum  # 更新最小距离和
                                                                min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                                                min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                                                min_distance_bot_id = bot_id  # 更新最小距离和下的机器人ID
                                                                min_distance_object_type = object_type  # 更新最小距离和下的物品类型
                                                # 如果不是8、9，要判断计划原材料格状态
                                                else:
                                                    # 遍历这种每个工作台
                                                    for table_id_2 in table_index[f'type_{table_type_id_2}']:
                                                        # 如果在工作区域内
                                                        if table_id_2 in bot_work_area[f'bot_{bot_id}']:
                                                            # 如果缺少这个材料
                                                            if object_type in table_list[table_id_2][
                                                                'projected_missing_material']:
                                                                distance_2 = calculate_distance_between_table(
                                                                    table_id_1,
                                                                    table_id_2)  # 计算购买工作台与出售工作台之间的距离
                                                                distance_sum = distance_1 + distance_2  # 计算两段距离和
                                                                # 如果当前距离和小于存的最小距离和
                                                                if distance_sum < min_distance_sum:
                                                                    min_distance_sum = distance_sum  # 更新最小距离和
                                                                    min_distance_table_id_1 = table_id_1  # 更新最小距离和下的第一个目标工作台ID
                                                                    min_distance_table_id_2 = table_id_2  # 更新最小距离和下的第二个目标工作台ID
                                                                    min_distance_bot_id = bot_id  # 更新最小距离和下的机器人ID
                                                                    min_distance_object_type = object_type  # 更新最小距离和下的物品类型
    # 如果找不到需要购买的产品或找到了要购买的产品但是没地方卖
    if min_distance_table_id_1 == -1 or min_distance_table_id_2 == -1:
        return 0  # 不做抢占
    # 发生抢占
    else:
        # 复原原计划购买工作台的计划产品格状态
        table_list[bot_list[min_distance_bot_id]['target_table_id'][0]]['projected_production'] = 1
        # 复原原计划出售工作台的计划原材料格状态
        table_list[bot_list[min_distance_bot_id]['target_table_id'][1]]['projected_missing_material'].append(
            table_list[bot_list[min_distance_bot_id]['target_table_id'][0]]['type'])
        # 新任务
        table_list[min_distance_table_id_1]['projected_production'] = 0  # 更新工作台的考虑计划的产品格状态
        # 如果出售工作台不是8、9，就要更新工作台的考虑计划的原料格状态
        if table_list[min_distance_table_id_2]['type'] not in [8, 9]:
            table_list[min_distance_table_id_2]['projected_missing_material'].remove(
                min_distance_object_type)  # 更新工作台的考虑计划的原料格状态
        bot_list[min_distance_bot_id]['target_table_id'] = (
            min_distance_table_id_1, min_distance_table_id_2)  # 把任务写到机器人字典中
        bot_list[min_distance_bot_id]['work_status'] = 1  # 把工作状态改为前往第一个目标工作台
        return 1  # 做出抢占


# def map_1_task_schedule(preemption_distance_kp, wait_threshold):
#     """
#     地图1机器人总体任务规划。
#     :param preemption_distance_kp: 抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
#     :param wait_threshold: 原地购买等待时间阈值
#     :return: none
#     """
#     global bot_list
#     # 抢占优先级
#     task_schedule_in_preempting_priority([7], preemption_distance_kp)
#     task_schedule_in_preempting_priority([4, 5, 6], preemption_distance_kp)
#     # 更新动态优先级调度计数器
#     dynamic_priority.counter_update()
#     # 遍历机器人
#     for bot_id in range(bot_num):
#         # 如果机器人空闲则规划任务
#         if bot_list[bot_id]['work_status'] == 0:
#             # 先判断原地是否能够买
#             # 可能存在抢占的情况
#             if task_schedule_by_inplace_buy(bot_id, wait_threshold):
#                 continue
#             # 动态优先级调度
#             if task_schedule_in_dynamic_priority(bot_id):
#                 continue
#             # 按从高到低的优先级遍历工作台是否有产品
#             # 工作台产品优先级为(7)(654)(321)
#             # 第一优先级(7)
#             if task_schedule_in_the_same_priority(bot_id, [7]):
#                 continue
#             # 第二优先级(456)
#             if task_schedule_in_the_same_priority(bot_id, [4, 5, 6]):
#                 continue
#             # 第三优先级(123)
#             if task_schedule_in_the_same_priority(bot_id, [1, 2, 3]):
#                 continue


def map_1_task_schedule(preemption_distance_kp, wait_threshold):
    """
    地图1机器人总体任务规划。
    :param preemption_distance_kp: 抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
    :param wait_threshold: 原地购买等待时间阈值
    :return: none
    """
    global bot_list
    # 抢占优先级
    task_schedule_in_preempting_priority_map_1_special([7], preemption_distance_kp)
    task_schedule_in_preempting_priority_map_1_special([4, 5, 6], preemption_distance_kp)
    # 更新动态优先级调度计数器
    dynamic_priority.counter_update()
    # 遍历机器人
    for bot_id in range(bot_num):
        # 如果机器人空闲则规划任务
        if bot_list[bot_id]['work_status'] == 0:
            # 先判断原地是否能够买
            # 可能存在抢占的情况
            if task_schedule_by_inplace_buy_map_1_special(bot_id, wait_threshold):
                continue
            # 动态优先级调度
            if task_schedule_in_dynamic_priority(bot_id):
                continue
            # 按从高到低的优先级遍历工作台是否有产品
            # 工作台产品优先级为(7)(654)(321)
            # 第一优先级(7)
            if task_schedule_in_the_same_priority_map_1_special(bot_id, [7]):
                continue
            # 第二优先级(456)
            if task_schedule_in_the_same_priority_map_1_special(bot_id, [4, 5, 6]):
                continue
            # 第三优先级(123)
            if task_schedule_in_the_same_priority_map_1_special(bot_id, [1, 2, 3]):
                continue


def map_2_task_schedule(preemption_distance_kp, wait_threshold):
    """
    地图2机器人总体任务规划。
    :param preemption_distance_kp: 抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
    :param wait_threshold: 原地购买等待时间阈值
    :return: none
    """
    global bot_list
    # 抢占优先级
    task_schedule_in_preempting_priority([7], preemption_distance_kp)
    task_schedule_in_preempting_priority([4, 5, 6], preemption_distance_kp)
    # 更新动态优先级调度计数器
    dynamic_priority.counter_update()
    # 遍历机器人
    for bot_id in range(bot_num):
        # 如果机器人空闲则规划任务
        if bot_list[bot_id]['work_status'] == 0:
            # 先判断原地是否能够买
            # 可能存在抢占的情况
            if task_schedule_by_inplace_buy(bot_id, wait_threshold):
                continue
            # 动态优先级调度
            if task_schedule_in_dynamic_priority(bot_id):
                continue
            # 按从高到低的优先级遍历工作台是否有产品
            # 工作台产品优先级为(7)(654)(321)
            # 第一优先级(7)
            if task_schedule_in_the_same_priority(bot_id, [7]):
                continue
            # 第二优先级(456)
            if task_schedule_in_the_same_priority(bot_id, [4, 5, 6]):
                continue
            # 第三优先级(123)
            if task_schedule_in_the_same_priority(bot_id, [1, 2, 3]):
                continue


# def map_3_task_schedule(preemption_distance_kp, wait_threshold):
#     """
#     地图3机器人总体任务规划。
#     :param preemption_distance_kp: 抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
#     :param wait_threshold: 原地购买等待时间阈值
#     :return: none
#     """
#     global bot_list
#     # 抢占优先级
#     task_schedule_in_preempting_priority([7], preemption_distance_kp)
#     task_schedule_in_preempting_priority([4, 5, 6], preemption_distance_kp)
#     # 更新动态优先级调度计数器
#     # dynamic_priority.counter_update()
#     # 遍历机器人
#     for bot_id in range(bot_num):
#         # 如果机器人空闲则规划任务
#         if bot_list[bot_id]['work_status'] == 0:
#             # 先判断原地是否能够买
#             # 可能存在抢占的情况
#             if task_schedule_by_inplace_buy(bot_id, wait_threshold):
#                 continue
#             # 动态优先级调度
#             # if task_schedule_in_dynamic_priority(bot_id):
#             #     continue
#             # 按从高到低的优先级遍历工作台是否有产品
#             # 工作台产品优先级为(7)(654)(321)
#             # 第一优先级(7)
#             if task_schedule_in_the_same_priority(bot_id, [7]):
#                 continue
#             # 第二优先级(456)
#             if task_schedule_in_the_same_priority(bot_id, [4, 5, 6]):
#                 continue
#             # 第三优先级(123)
#             if task_schedule_in_the_same_priority(bot_id, [1, 2, 3]):
#                 continue


def map_3_task_schedule(preemption_distance_kp, wait_threshold, bot_work_area):
    """
    地图3机器人总体任务规划。
    :param preemption_distance_kp: 抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
    :param wait_threshold: 原地购买等待时间阈值
    :param bot_work_area: 机器人工作区域字典
    :return: none
    """
    global bot_list
    # 抢占优先级
    task_schedule_in_preempting_priority_map_3_special([7], bot_work_area, preemption_distance_kp)
    task_schedule_in_preempting_priority_map_3_special([4, 5, 6], bot_work_area, preemption_distance_kp)
    # 更新动态优先级调度计数器
    # dynamic_priority.counter_update()
    # 遍历机器人
    for bot_id in range(bot_num):
        # 如果机器人空闲则规划任务
        if bot_list[bot_id]['work_status'] == 0:
            # 先判断原地是否能够买
            # 可能存在抢占的情况
            if task_schedule_by_inplace_buy_map_3_special(bot_id, bot_work_area, wait_threshold):
                continue
            # 动态优先级调度
            # if task_schedule_in_dynamic_priority(bot_id):
            #     continue
            # 按从高到低的优先级遍历工作台是否有产品
            # 工作台产品优先级为(7)(654)(321)
            # 第一优先级(7)
            if task_schedule_in_the_same_priority_map_3_special(bot_id, bot_work_area, [7]):
                continue
            # 第二优先级(456)
            if task_schedule_in_the_same_priority_map_3_special(bot_id, bot_work_area, [4, 5, 6]):
                continue
            # 第三优先级(123)
            if task_schedule_in_the_same_priority_map_3_special(bot_id, bot_work_area, [1, 2, 3]):
                continue


def map_4_task_schedule(preemption_distance_kp, wait_threshold):
    """
    地图1机器人总体任务规划。
    :return: none
    """
    global bot_list
    # 抢占优先级
    task_schedule_in_preempting_priority([7], preemption_distance_kp)
    task_schedule_in_preempting_priority([4, 5, 6], preemption_distance_kp)
    # 更新动态优先级调度计数器
    dynamic_priority.counter_update()
    # 遍历机器人
    for bot_id in range(bot_num):
        # 如果机器人空闲则规划任务
        if bot_list[bot_id]['work_status'] == 0:
            # 先判断原地是否能够买
            # 可能存在抢占的情况
            if task_schedule_by_inplace_buy(bot_id, wait_threshold):
                continue
            # 动态优先级调度
            if task_schedule_in_dynamic_priority(bot_id):
                continue
            # 按从高到低的优先级遍历工作台是否有产品
            # 工作台产品优先级为(7)(654)(321)
            # 第一优先级(7)
            if task_schedule_in_the_same_priority(bot_id, [7]):
                continue
            # 第二优先级(456)
            if task_schedule_in_the_same_priority(bot_id, [4, 5, 6]):
                continue
            # 第三优先级(123)
            if task_schedule_in_the_same_priority(bot_id, [1, 2, 3]):
                continue


def map_1_run():
    """
    地图1运行函数。
    :return: none
    """
    # 参数
    preemption_distance_kp = 0.6  # 抢占优先级参数，抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
    wait_threshold = 50  # 原地购买参数，最长等待时间
    min_distance_kp = 1  # 防碰撞参数，最小避障距离系数，距离大于圆心距乘该系数才触发避障
    max_distance_kp = 10  # 防碰撞参数，最大避障距离系数，距离小于圆心距乘该系数才触发避障
    while True:
        try:
            # 更新帧数据
            update_data()
            # 任务规划
            map_1_task_schedule(preemption_distance_kp, wait_threshold)
            # 控制
            control()
            # 防碰撞
            anti_collision(min_distance_kp, max_distance_kp)
            # 发送指令
            send_data()
            # 调试用
            get_max_money()
        except EOFError:  # 读到9000帧后下一帧是EOF,此时停止程序
            # 调试用
            logging.info(max_money)
            break


def map_2_run():
    """
    地图2运行函数。
    :return: none
    """
    # 参数
    preemption_distance_kp = 0.6  # 抢占优先级参数，抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
    wait_threshold = 100  # 原地购买参数，最长等待时间
    min_distance_kp = 1  # 防碰撞参数，最小避障距离系数，距离大于圆心距乘该系数才触发避障
    max_distance_kp = 5  # 防碰撞参数，最大避障距离系数，距离小于圆心距乘该系数才触发避障
    while True:
        try:
            # 更新帧数据
            update_data()
            # 任务规划
            map_2_task_schedule(preemption_distance_kp, wait_threshold)
            # 控制
            control()
            # 防碰撞
            anti_collision(min_distance_kp, max_distance_kp)
            # 发送指令
            send_data()
            # 调试用
            get_max_money()
        except EOFError:  # 读到9000帧后下一帧是EOF,此时停止程序
            # 调试用
            logging.info(max_money)
            break


def map_3_run():
    """
    地图3运行函数。
    :return: none
    """
    # 参数
    preemption_distance_kp = 0.6  # 抢占优先级参数，抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
    wait_threshold = 10  # 原地购买参数，最长等待时间
    min_distance_kp = 1  # 防碰撞参数，最小避障距离系数，距离大于圆心距乘该系数才触发避障
    max_distance_kp = 5  # 防碰撞参数，最大避障距离系数，距离小于圆心距乘该系数才触发避障
    # 机器人工作区域
    bot_work_area = {
        'bot_0': [0, 1, 2, 3, 4, 5, 6, 7, 24],
        'bot_1': [8, 9, 10, 15, 17, 18, 19, 26, 27, 30, 31, 24],
        'bot_2': [11, 12, 13, 14, 16, 20, 21, 22, 23, 25, 28, 29, 32, 33, 34, 24],
        'bot_3': [35, 36, 41, 42, 44, 45, 47, 48, 24],
    }
    while True:
        try:
            # 更新帧数据
            update_data()
            # 任务规划
            map_3_task_schedule(preemption_distance_kp, wait_threshold, bot_work_area)
            # 控制
            control()
            # 防碰撞
            anti_collision(min_distance_kp, max_distance_kp)
            # 发送指令
            send_data()
            # 调试用
            get_max_money()
        except EOFError:  # 读到9000帧后下一帧是EOF,此时停止程序
            # 调试用
            logging.info(max_money)
            break


def map_4_run():
    """
    地图4运行函数。
    :return: none
    """
    # 参数
    preemption_distance_kp = 0.6  # 抢占优先级参数，抢占购买距离系数，当原购买任务剩余距离大于该系数乘待抢占购买任务距离时就被抢占
    wait_threshold = 10  # 原地购买参数，最长等待时间
    min_distance_kp = 1.4  # 防碰撞参数，最小避障距离系数，距离大于圆心距乘该系数才触发避障
    max_distance_kp = 100  # 防碰撞参数，最大避障距离系数，距离小于圆心距乘该系数才触发避障
    while True:
        try:
            # 更新帧数据
            update_data()
            # 任务规划
            map_4_task_schedule(preemption_distance_kp, wait_threshold)
            # 控制
            control()
            # 防碰撞
            anti_collision(min_distance_kp, max_distance_kp)
            # 发送指令
            send_data()
            # 调试用
            get_max_money()
        except EOFError:  # 读到9000帧后下一帧是EOF,此时停止程序
            # 调试用
            logging.info(max_money)
            break


def run():
    """
    主体运行函数。
    :return: none
    """
    global map_num
    if map_num == 1:
        map_1_run()
    elif map_num == 2:
        map_2_run()
    elif map_num == 3:
        map_3_run()
    elif map_num == 4:
        map_4_run()


# endregion


# region 调试用的函数
def get_max_money():
    """
    获取最大金钱数。
    :return: none
    """
    global money, max_money
    if money > max_money:
        max_money = money


# endregion


if __name__ == '__main__':
    # 常数
    bot_num = 4  # 机器人数量
    max_forward_velocity = 6  # 最大前进速度
    min_forward_velocity = 1  # 最小前进速度
    max_rotate_velocity = numpy.pi  # 最大旋转速度
    min_rotate_velocity = 1  # 最小启动角速度
    no_load_radius = 0.45  # 机器人无载半径
    on_load_radius = 0.53  # 机器人有载半径
    # 数据储存
    frame_id = 0  # 初始化帧id
    money = 200000  # 初始化金钱数
    table_num = 0  # 初始化工作台数量
    bot_list = []  # 初始化机器人列表
    table_list = []  # 初始化工作台列表
    table_type = []  # 初始化工作台类别列表
    table_index = {}  # 初始化工作台索引字典
    map_num = 0  # 地图编号
    # 指令字典
    command = {
        'forward': [0, 0, 0, 0],  # 参数为四个机器人速度值
        'rotate': [0, 0, 0, 0],  # 参数为四个机器人角速度值
        'buy': [],  # 参数为机器人ID
        'sell': [],  # 参数为机器人ID
        'destroy': [],  # 参数为机器人ID
    }
    # 动态优先级调度对象
    dynamic_priority = DynamicPrioritySchedule()

    # 调试用的
    max_money = 200000  # 最大金钱数

    # get_init_data(local_test())    # 本地调试用这个
    get_init_data(judge_machine())  # 在用判题器的时候就启用这个
    # get_map_num()  # 获取地图编号
    logging.info(map_num)

    run()

    # while True:
    #     try:
    #         # 更新帧数据
    #         update_data()
    #         # 任务规划
    #         task_schedule()
    #         # 控制
    #         control()
    #
    #         # 碰撞预测
    #         anti_collision(1.4, 100)
    #
    #         # 停止工作
    #         # stop_work()
    #
    #         # 发送指令
    #         send_data()
    #
    #         # 调试用
    #         get_max_money()
    #
    #         # if 7500 < frame_id < 8500:
    #         #     logging.info(frame_id)
    #         #     logging.info(f'1 {bot_list[1]["target_table_id"]} {bot_list[1]["work_status"]}')
    #         #     logging.info(f'2 {bot_list[2]["target_table_id"]} {bot_list[2]["work_status"]}')
    #     except EOFError:  # 读到9000帧后下一帧是EOF,此时停止程序
    #
    #         # 调试用
    #         logging.info(max_money)
    #
    #         break
