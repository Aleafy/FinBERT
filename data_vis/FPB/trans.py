import json

data_list = json.load(open('/home/fangye/FinSA/data_vis/FPB/train_ori.json','r'))

new_list = []

for data in data_list:
    new_data = {}
    new_data['sentence'] = data['sentence']
    new_data['label'] = (data['label']+1)%3

    new_list.append(new_data)

print(len(new_list))
print(len(data_list))


json.dump(new_list, open('/home/fangye/FinSA/data_vis/FPB/train.json','w'), indent=1)
