import os
import pandas as pd

path_data_all = "data/data_all_fangji.csv"  # 原方剂数据集
path_ner_result = "result/ner_predict_test.utf8"    # 命名实体识别结果
path_result = "result/result.csv"   # 存储被识别出来的实体的标准表示
path_data_all_ner = "result/data_all_ner.csv"   # 将命名实体识别结果写到原方剂数据集中
path_entity_new = "result/entity_new.txt"  # 存储被识别出来的原词库中不存在的实体（后期进行人工审核，统计其中被发现的新词）


def get_data():
    """
    从命名实体识别结果中获取标准的实体词
    :return:
    """
    with open(path_ner_result, "r", encoding="utf-8") as f_ner:
        entity = ""  # 用于保存一个实体
        entity_list = []    # 用于保存一个方剂的所有实体
        entity_all = []  # 用于保存所有方剂的实体
        lines = f_ner.readlines()
        for line in lines:
            char_tag_predict_list = line.split()
            # print("char_tag_predict_list:", char_tag_predict_list)
            if len(char_tag_predict_list) == 0:
                # print("entity_list:", entity_list)
                # 每个子列表拼接成字符串，否则若直接输出列表到文件，导致后续读取数据不方便，另外，用join拼接字符串更为方便
                entity_all.append("、".join(entity_list))
                # print("entity_all:", entity_all)
                entity_list = []
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
                    entity_list.append(entity)
                    entity = ""
    print("entity_all:", entity_all)
    entity_all_series = pd.Series(entity_all, name="NER_result")
    # print("entity_all_series:", entity_all_series)
    entity_all_series = pd.DataFrame(entity_all_series)  # 转换成DataFrame才能将列名name输入到csv中
    # print("entity_all_series:", entity_all_series)
    entity_all_series.to_csv(path_result, index=False, encoding="utf-8")


def find_new_entity():
    """
    利用集合的差集，寻找识别出的实体中未出现在原实体库中的新实体，用以后续的研究
    :return:
    """
    sets_name_list = ["diseases", "pattern", "treat", "symptom"]    # 四个原实体词库的文件名（病名、证型、治疗手段、症状）
    set_old = set()  # 用于保存原实体词库中的词库
    for set_name in sets_name_list:
        path_set = os.path.join("data", set_name+".txt")
        with open(path_set, "r", encoding="utf-8") as f_set:
            lines = f_set.readlines()
            for line in lines:
                entity = line.strip()
                set_old.add(entity)
    print("set_old:", set_old)
    set_ner = set()  # 用于保存算法识别出的实体词
    entity_ner_all = pd.read_csv(path_result)
    for i in range(entity_ner_all.shape[0]):
        entity_ner_list = entity_ner_all["NER_result"].loc[i].split("、")
        for entity in entity_ner_list:
            set_ner.add(entity)
    print("set_ner:", set_ner)
    set_new = set_ner - set_old  # 得到两个集合的差集（set_ner中set_old中未出现的新实体）
    print("set_new:", set_new)
    with open(path_entity_new, "w", encoding="utf-8") as f_new:  # 将新实体写入txt文件
        list_new = list(set_new)
        entity_string = "\n".join(list_new)
        f_new.write(entity_string)


def write_to_data():
    """
    将NER算法的实体识别结果写入到原方剂数据集中
    :return:
    """
    data_all = pd.read_csv(path_data_all)
    result = pd.read_csv(path_result)
    # print(result)
    print(result.info())
    data_all.insert(11, "NER_result", None)
    data_all["NER_result"] = result["NER_result"]
    print(data_all.info())
    data_all.to_csv(path_data_all_ner, encoding="utf-8")


if __name__ == "__main__":
    get_data()
    find_new_entity()
    write_to_data()
