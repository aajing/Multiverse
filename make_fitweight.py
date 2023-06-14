from collections import OrderedDict
import os
import torch
import cfg.config as cfg
def copyStateDict(state_dict):
    print("copydict!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        if k.endswith('mask'):
            continue
        new_state_dict[k] = v
    print("end copydict!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    return new_state_dict
filenames = []
results_path = cfg.base_dir + "/result_global" 
for strategy in [ 'GlobalMagWeight']:
        for c in [1,10/9, 10/8,10/7,10/6, 2, 10/4, 10/3, 5, 10]:
            # for epoch in [0,3,5,10]:
            for epoch in [10]:
            # for i in [5]:
                position = results_path+'/'+ strategy+'/'+str(c)+'-'+str(epoch)
                file = os.listdir(position)
                position = position+'/'+file[0]
                filelist = os.listdir(position +'/'+'checkpoints')
                filelist.sort()
                num = len(filelist)
                if num == 1:
                    filename = position +'/'+'checkpoints' + '/'+filelist[0]
                elif num==2:
                    filename = position +'/'+'checkpoints' + '/'+filelist[1]
                else:
                    continue

                filenames.append(filename)
print(filenames)

for file in filenames:

    model = torch.load(file)
    # for i in model:
    #     print(i)
    state_dict = model['model_state_dict']
    new_dict=copyStateDict(state_dict)

    torch.save(new_dict,file.rsplit('-',1)[0]+'-weuse.pt')