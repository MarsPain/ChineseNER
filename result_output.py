import os

path_data_all = "data/data_all_fangji.csv"  # 原方剂数据集
path_ner_result = "result/ner_predict_test.utf8"    # 命名实体识别结果
path_result = "result/result.txt"   # 存储被识别出来的实体的标准表示
path_entity_new = "result/entity_new.txt"  # 存储被识别出来的原词库中不存在的实体（后期进行人工审核，统计其中被发现的新词）


def get_data():
    """
    从命名实体识别结果中获取标准的实体词
    :param ner_result:命名实体识别结果的文件
    :return:
    """
    with open(path_ner_result, "r", encoding="utf-8") as f_ner:
        entity = ""  # 用于保存一个实体
        string_entity = []    # 用于保存一个方剂的实体
        string = ""  # 用于保存所有实体
        lines = f_ner.readlines()
        for line in lines:
            char_tag_predict_list = line.split()
            # print("char_tag_predict_list:", char_tag_predict_list)
            if len(char_tag_predict_list) == 0:
                # print("string_entity:", string_entity)
                string = string + "、".join(string_entity) + "\n"    # 用join拼接字符串更为方便
                # print("string:", string)
                string_entity = []
            elif char_tag_predict_list[-1] == "O":
                continue
            else:
                char = char_tag_predict_list[0]  # 被标注的字符
                predict = char_tag_predict_list[-1]  # NER模型的预测
                predict_loc = predict[0]   # 字符在实体中的位置(B/I/E)
                if predict_loc == "B" or predict_loc == "I":
                    # print("entity:", entity)
                    entity += char
                elif predict_loc == "E":
                    entity += char
                    string_entity.append(entity)
                    entity = ""
    # print(string)
    with open(path_result, "w", encoding="utf-8") as f:
        f.write(string)


def find_new_entity():
    """
    利用集合的差集，寻找识别出的实体中未出现在原实体库中的新实体，用以后续的研究
    :return:
    """
    sets_name_list = ["pattern", "symptom", "treat", "diseases"]    # 四个原实体词库的文件名
    set_old = set()  # 用于保存原实体词库中的词库
    for set_name in sets_name_list:
        path_set = os.path.join("data", set_name+".txt")
        with open(path_set, "r", encoding="utf-8") as f_set:
            lines = f_set.readlines()
            for line in lines:
                entity = line.strip()
                set_old.add(entity)
    print("set_old:", set_old)
    set_ner = set()
    with open(path_result, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            entity_list = line.strip().split("、")
            # print(entity_list)
            for entity in entity_list:
                set_ner.add(entity)
    print("set_ner:", set_ner)
    set_new = set_ner - set_old  # 得到两个集合的差集（set_ner中set_old中未出现的新实体）
    print("set_new:", set_new)
    with open(path_entity_new, "w", encoding="utf-8") as f_new:  # 将新实体写入txt文件
        list_new = list(set_new)
        entity_string = "\n".join(list_new)
        f_new.write(entity_string)


# 将实体词写入原方剂数据集中
def write_to_data():
    pass


if __name__ == "__main__":
    # get_data()
    # find_new_entity()
    write_to_data()
