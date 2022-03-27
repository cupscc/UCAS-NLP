import json
import re
title = ''
list = []
with open(r"E:\2022spring\nlp\word\mytime\renmin.json","r",encoding="utf-8") as f:#输入json文件
    with open(r"E:\2022spring\nlp\word\mytime\renmin_processed.json","w",encoding="utf-8") as fout:#输出json文件
        json_data = json.load(f)#加载json数据
        for i in range(len(json_data)):
            title = json_data[i]['title']
            result = ''
            for j in range(len(json_data[i]['contents'])):
                replace_temp = json_data[i]['contents'][j]
                replace_result = ''.join(re.findall(u'[\u4e00-\u9fff]+', replace_temp))#筛选所有中文汉字
                result = result + replace_result#把分段的文章拼接起来
            list.append({'title':title,'result':result})
        json.dump(list,fout,sort_keys=True,ensure_ascii=False)#把json加载到新的处理过的文档