import os

path_ner_result = "result/ner_predict_test.utf8"
path_result = "result/result.csv"


def output_data(ner_result, result):
    with open(ner_result, "r", encoding="utf-8") as f_ner:
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
    print(string)

if __name__ == "__main__":
    output_data(path_ner_result, path_result)


def output_data_beifen(model_name, path_model_name):
    filename_predict = os.path.join("result", model_name+"_predict.utf8")
    filename_result = os.path.join("result", model_name+".txt")
    # filename_predict = os.path.join("resultB", model_name+"_predict.utf8")
    # filename_result = os.path.join("resultB", model_name+".txt")
    if model_name == "zengjianchi":
        result = "公告id	股东全称	股东简称	变动截止日期	变动价格	变动数量	变动后持股数	变动后持股比例"
    elif model_name == "hetong":
        result = "公告id	甲方	乙方	项目名称	合同名称	合同金额上限	合同金额下限	联合体成员"
    elif model_name == "dingzeng":
        result = "公告id  增发对象    增发数量	增发金额	锁定期	认购方式"
    elif model_name == "test":
        result = ""
    with open(filename_predict, "r", encoding="utf-8") as f:
        flag = True
        temp_result = ["\t" for i in range(8)]   # 用于保存一行结构化的实体
        temp2_result = []    # 保存临时重复出现的实体
        entity = "" # 用于保存一个实体
        for line in f:
            # print(line)
            char_tag_predict = line.split()
            # print(char_tag_predict)
            # print(len(char_tag_predict))
            if len(char_tag_predict) != 3:
                continue
            if char_tag_predict[-1] == "O":
                # print(char_tag_predict[0])
                continue
            else:
                predict = char_tag_predict[-1]
                char = char_tag_predict[0]  # 被标注预测的字符
                predict_loc = predict[0]    # 被标注预测的在实体中的位置（B、I、O等）
                predict_type = predict[-1]  # 被标注预测的在实体类型（0-7，标注还是得从0而不是1开始，方便输出）
                # print(type(predict_type))
                # print("char:", char, "predict_loc:", predict_loc, "predict_type:", predict_type)
                if predict_loc == "B" or predict_loc == "I":
                    entity += char
                elif predict_loc == "E":
                    entity += char
                    # 如果是新的公告id实体，则说明属于新的公告文本，将上一个temp_result加入result并换行，
                    # 然后重新初始化temp_result并用entity对公告id赋值。
                    if int(predict_type) == 1:
                        result = result + "\t".join(temp_result) + "\n"
                        temp_result = ["\t" for i in range(8)]
                        temp_result[0] = entity
                        entity = ""
                    # 如果是当前公告中的一条结构化信息，则只要对该行数据的temp_result赋值即可
                    elif temp_result[int(predict_type)-1] == "\t":
                        temp_result[int(predict_type)-1] = entity
                        entity = ""
                    # 该实体依然在当前的公告文本中，但是属于另一条结构化信息，所以先从当前temp_result
                    # 中取出公告id，然后将temp_result加入result中，再换行，
                    # 重新初始化temp_result，并对公告id和当前的检测到的实体entity赋值
                    # 这个地方还是没处理好，很多不应该存在的多余的结构化数据，而且导致一些实体数据被分散在多余的结构化数据中。
                    # 有两个思路，第一个思路是在遍历到下一个公告之前，同一个公告的信息用嵌套列表存储，然后重复出现的实体依次往后放
                    # 在下一个子列表中，然后添加实体的时候从最前面的列表开始检索相应位置是否是空着的，如果空着的就填入，最后把长度不够的子列表去除！
                    # 更好的思路是用一个临时列表temp2_result保存重复的实体，不重复的实体先放在当前temp_result，当temp2_result中非空的元素长度与
                    # temp_result相同时（根据规律，文本中出现的重复结构化数据多为列表表示，所以实体数量基本相等），
                    # 则认为是属于同一个公告的结构化数据,temp_result添加到最终result中，temp2_result中的值转移到temp_result中，
                    # 然后自身赋值为空数组
                    # elif temp_result[int(predict_type)-1] == "\t":
                    #     if temp2_result == []:  #如果出现了重复实体且temp2_result为空
                    #         count1, count2 =0, 0
                    #         announce_id = temp_result[0]
                    #         temp2_result = ["\t" for i in range(8)]
                    #         temp2_result[0] = announce_id
                    #         temp2_result[int(predict_type)-1] = entity
                    #         entity = ""
                    #     else:
                    #         if temp2_result[int(predict_type)-1] == "\t":
                    #             temp2_result[int(predict_type)-1] = entity
                    #         entity = ""  #如果出现的重复实体在临时列表中已经重复了，则忽略
                    #     for e in temp_result:
                    #         if e != "\t":
                    #             count1 += 1
                    #     for e in temp2_result:
                    #         if e != "\t":
                    #             count2 += 1
                    #     if count1 == count2:    #如果实体数量相同了
                    #         result = result + "\t".join(temp2_result) + "\n"
                    #         temp2_result = []

                    # 下面的方法是简单粗暴地出现重复的公司名称时就认为是另一条结构化数据
                    # elif int(predict_type) == 2 and temp_result[1] != "\t":
                    #     announce_id = temp_result[0]
                    #     result = result + "\t".join(temp_result) + "\n"
                    #     temp_result = ["\t" for i in range(8)]
                    #     temp_result[0] = announce_id
                    #     temp_result[int(predict_type)-1] = entity
                    #     entity = ""
                    # else:
                    #     entity = ""

                    # 下面的方法是简单粗暴地认为出现重复位置的实体就认为是另一条结构化数据,
                    # 出现之后直接换一行，而且从上一个结构化信息继承新一行缺失的数据（这种确实大概率是因为共享同样的实体）。
                    elif temp_result[int(predict_type)-1] != "\t":
                        new_temp_result = ["\t" for i in range(8)]
                        for i in range(int(predict_type)-1):
                            new_temp_result[i] = temp_result[i]
                        new_temp_result[int(predict_type)-1] = entity
                        result = result + "\t".join(temp_result) + "\n"
                        temp_result = new_temp_result
                        new_temp_result = ["\t" for i in range(8)]
                        entity = ""

    with open(filename_result, "w", encoding="utf-8") as f:
        f.write(result)
