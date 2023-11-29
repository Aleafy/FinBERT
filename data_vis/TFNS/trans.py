import json

data_list = json.load(open('/home/fangye/FinSA/data_vis/TFNS/test_ori.json','r'))

new_list = []

for data in data_list:
    new_data = {}
    new_data['sentence'] = data['text']
    new_data['label'] = (2*data['label']+1)%3

    new_list.append(new_data)

print(len(new_list))
print(len(data_list))


json.dump(new_list, open('/home/fangye/FinSA/data_vis/TFNS/test.json','w'), indent=1)
