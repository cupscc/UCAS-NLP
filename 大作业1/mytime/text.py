import json
with open(r"E:\2022spring\nlp\word\mytime\renmin_processed.json","r",encoding="utf-8") as fin:
    json_data = json.load(f)  # 加载json数据
    print(len(json_data))