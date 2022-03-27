import json
import math
import matplotlib.pyplot as plt
def computing_entropy(all_character,character_number,times):
     all_likelihood = {}
     for k in all_character.keys():
          all_likelihood[k] = all_character[k]/character_number
     entropy = 0
     for k in all_likelihood.keys():
          single = (all_likelihood[k]) * math.log2(all_likelihood[k])
          entropy = entropy - single
     print("time",times,"text length:",character_number," entropy:",entropy,"\n",end='')
     return [character_number,entropy]

with open(r"E:\2022spring\nlp\word\mytime\renmin_processed.json","r",encoding="utf-8") as fin:
     all_character = {}
     all_cha_num = 0
     json_data = json.load(fin)
     #print(type(json_data))
     all_entry = len(json_data)
     list_x = []
     list_y = []
     for i in range(15):
          for j in range(2000):
               processing_index = j + i * 2000
               processing_string = json_data[processing_index]['result']
               processing_length = len(processing_string)
               all_cha_num += processing_length
               for k in range (processing_length):
                    if processing_string[k] in all_character:
                         all_character[processing_string[k]] += 1
                    else:
                         all_character[processing_string[k]] = 1
          res = computing_entropy(all_character,all_cha_num,i+1)
          list_x.append(res[0])
          list_y.append(res[1])
     # trace = plotly.graph_objs.Scatter(
     #      x = list_x,
     #      y = list_y,
     #      mode="lines",
     #      name="res",
     #      line = dict(
     #           color = 'rgba(255,182,193)',
     #           width = 1
     #      )
     # )
     plt.plot(list_x,list_y)
     plt.ylim((9,10))
     plt.show()
     #plotly.offline.init_notebook_mode()
     #plotly.offline.iplot(fig,filename='happy')





