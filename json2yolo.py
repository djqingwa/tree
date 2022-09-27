
import json
import os
encoding="unicode_escape"
name2id = {'弓箭手':0,'小胖':1,'主角':2,'小兵':3,'boss':4}
               
def convert(img_size, box):
    dw = 1./(img_size[0])
    dh = 1./(img_size[1])
    x = (box[0] + box[2])/2.0 - 1
    y = (box[1] + box[3])/2.0 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)
 
 
def decode_json(json_floder_path,json_name):
    #转换好的标签路径放哪里
    txt_name = 'data\\custom\\labels\\' + json_name[0:-5] + '.txt'
    txt_file = open(txt_name, 'w')#打开文件往这里去写
 
    json_path = os.path.join(json_floder_path, json_name)
    #data = json.load(open(json_path, 'r',encoding="utf-8"))
    data = json.load(open(json_path, 'rb'))

    img_w = data['imageWidth']
    img_h = data['imageHeight']
 
    for i in data['shapes']:
        
        label_name = i['label']
        if (i['shape_type'] == 'rectangle'):
 
            x1 = int(i['points'][0][0])
            y1 = int(i['points'][0][1])
            x2 = int(i['points'][1][0])
            y2 = int(i['points'][1][1])
 
            bb = (x1,y1,x2,y2)
            bbox = convert((img_w,img_h),bb)
            txt_file.write(str(name2id[label_name]) + " " + " ".join([str(a) for a in bbox]) + '\n')
    
if __name__ == "__main__":
    
    json_floder_path = 'myLableMe'#labelme生成标签的路径
    json_names = os.listdir(json_floder_path)#遍历有多少个json文件 
    for json_name in json_names:
        decode_json(json_floder_path,json_name)
