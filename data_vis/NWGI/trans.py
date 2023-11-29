import json

data_list = json.load(open('/home/fangye/FinSA/data_vis/NWGI/train_ori.json','r'))

new_list = []

for data in data_list:
    new_data = {}
    new_data['sentence'] = data['news']
    if 'positive' in data['label']:
        new_data['label'] = 0
    elif 'negative' in data['label']:
        new_data['label'] = 1
    elif 'neutral' in data['label']:
        new_data['label'] = 2

    new_list.append(new_data)

print(len(new_list))
print(len(data_list))


json.dump(new_list, open('/home/fangye/FinSA/data_vis/NWGI/train.json','w'), indent=1)
