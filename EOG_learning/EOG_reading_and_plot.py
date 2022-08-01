#%%
from array import array
from lib2to3.pgen2.pgen import generate_grammar
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

file_path = r'C:\Users\Administrator\Desktop\DATASET\S1\EOG.mat'
label_path = r'C:\Users\Administrator\Desktop\DATASET\S1\ControlSignal.mat'
mat_contents = sio.loadmat(file_path)
mat_labels = sio.loadmat(label_path)
print(mat_contents.keys())
print(mat_labels.keys())
# %%
raw_data = mat_contents['EOG']
labels = mat_labels['ControlSignal']
raw_data = np.array(raw_data)
labels = np.array(labels)
print(raw_data.shape, labels.shape)
# %%
# x = np.arange(0,raw_data.shape[1], 1)
x = np.arange(0,10000, 1)
y = raw_data[0,:10000]
fig, axs = plt.subplots()
axs.set_title('EOG signals')
axs.plot(x, y,color= 'C0')

for i in range(10000):
    if(labels[0,i] == 3):
        axs.scatter(x[i], y[i], color='r')
    if(labels[0,i] == 1):
        axs.scatter(x[i], y[i], color='y')
    if(labels[0,i] == 2):
        axs.scatter(x[i], y[i], color='g')

axs.set_xlabel('Time')
axs.set_ylabel('Amplitude')
plt.show()
# %%
x = np.arange(0,5000, 1)
y = raw_data[1,:5000]
fig, axs = plt.subplots()
axs.set_title('EOG signals')
axs.plot(x, y,color= 'C0')

for i in range(5000):
    if(labels[0,i] == 3):
        axs.scatter(x[i], y[i], color='r')

axs.set_xlabel('Time')
axs.set_ylabel('Amplitude')
plt.show()
# %%
blinks = []
idels = []
single_blink = []
single_idel = []
for i in range(raw_data.shape[1]-1):
    if(labels[0,i] == 3 and labels[0, i-1] == 3):
        single_blink.append(raw_data[1,i])
    if(labels[0,i] != 3 and labels[0, i-1] != 3):
        single_idel.append(raw_data[1,i])
    if(i!=0 and labels[0,i] == 3 and labels[0, i+1] != 3):
        blinks.append(single_blink)
        single_blink = []  
    if(i!=0 and labels[0,i] != 3 and labels[0, i+1] == 3):
        idels.append(single_idel)  
        single_idel = []
print(len(blinks), len(idels))
#%%
for i in range(len(blinks)):
    print(len(blinks[i]),len(idels[i]))
# %%
single_blink_x = np.arange(0,len(blinks[8]),1)
single_blink_y =  blinks[8]
single_idel_x = np.arange(0,len(idels[3]),1)
single_idel_y =  idels[3]
fig, axs = plt.subplots(2)
fig.suptitle('Singel trial of blink and idel')
axs[0].plot(single_blink_x, single_blink_y)
axs[1].plot(single_idel_x, single_idel_y)

# %%
print(np.var(single_blink_y),np.var(single_idel_y))
for i in range(len(idels)):
    idels[i] = idels[i] - np.mean(idels[i])
    blinks[i] = blinks[i] - np.mean(idels[i]) - np.mean(blinks[i])
#%%
import random
double_blink_rate = 0.2
genereted_signal = []
starts = []
ends = []
for i in range(80):
    flag = random.uniform(0,1)
    print(flag)
    if(flag > double_blink_rate):
        index = random.randint(0, len(idels)-1)
        for j in range(len(idels[index])):
            genereted_signal.append(idels[index][j])
    if(flag<=double_blink_rate):
        starts.append(len(genereted_signal))
        index = random.randint(0, len(blinks)-1)
        index2 = random.randint(0, len(blinks)-1)
        print(index, index2)
        for j in range(200):
            genereted_signal.append(blinks[index][j])
        for k in range(200):
            genereted_signal.append(blinks[index2][k])
        ends.append(len(genereted_signal))
        index_1 = random.randint(0, len(idels)-1)
        for j in range(600):
            if(j>=400):
                genereted_signal.append(idels[index_1][j-400])
            else:
                genereted_signal.append(idels[index_1][j])
        index_2 = random.randint(0, len(idels)-1)
        print('---',len(idels[index_2]))
        for j in range(240):
            genereted_signal.append(idels[index_2][j])
genereted_signal = np.array(genereted_signal)
fig, axs = plt.subplots()
fig.suptitle('Generated signals')
x_generated_sig = np.arange(0,len(genereted_signal),1)
axs.plot(x_generated_sig, genereted_signal)
# %%
print(np.var(genereted_signal), np.mean(genereted_signal), genereted_signal.shape)
#%%
def standardlizeSig(data):
    base = np.mean(data)
    std = np.std(data)
    standardlized_data = (data-base)/std
    del base, std
    return standardlized_data
genereted_signal_std = standardlizeSig(genereted_signal)
#%%
np.save('EOG_online_fake_data_real.npy', genereted_signal)
np.save('EOG_online_fake_data_std.npy', genereted_signal_std)
np.save('EOG_starts.npy', np.array(starts))
np.save('EOG_ends.npy', np.array(ends))

# %%
genereted_signal = np.array(genereted_signal)
blink_signals_blocks = []
idel_signal_blocks = []
for i,time_point in enumerate(starts):
    if(i == 0):
        for j in range(int((time_point-400)/50)):
            idel_signal_blocks.append(genereted_signal[j*50 :(j+1) * 50+350])
    if(i != 0):
        for j in range(int((starts[i]-ends[i-1]-400)/50)):
            idel_signal_blocks.append(genereted_signal[ends[i-1] + j* 50 :ends[i-1] + (j+1) * 50 +350])
for i in range(len(starts)):
    blink_signals_blocks.append(genereted_signal[starts[i]:ends[i]])
    
blink_signals_blocks = np.array(blink_signals_blocks)
idel_signal_blocks = np.array(idel_signal_blocks)
print(blink_signals_blocks.shape, idel_signal_blocks.shape)
# print(idel_signal_blocks.shape)
# %%
x = np.arange(0,400, 1)
y = idel_signal_blocks[103]
fig, axs = plt.subplots()
axs.set_title('Idel signal have a look')
axs.plot(x, y,color= 'C0')
axs.set_xlabel('Time')
axs.set_ylabel('Amplitude')
plt.show()
#%%
print(idel_signal_blocks.shape)
# %%
features_idel = np.array([np.std(idel_signal_blocks,axis=1), np.mean(idel_signal_blocks,axis=1), np.min(idel_signal_blocks,axis=1), np.max(idel_signal_blocks,axis=1)])
print(features_idel.shape)
labels_idel = np.zeros((features_idel[0].shape[0],1))
features_blink = np.array([np.std(blink_signals_blocks,axis=1), np.mean(blink_signals_blocks,axis=1), np.min(blink_signals_blocks,axis=1), np.max(blink_signals_blocks,axis=1)])
print(features_blink.shape)
labels_blink = np.ones((features_blink[0].shape[0],1))

index_blink = random.sample(range(0,features_blink.shape[1]),int(features_blink.shape[1]*0.2))
index_idel = random.sample(range(0,features_idel.shape[1]),int(features_idel.shape[1]*0.2))
test_data, test_labels = np.concatenate((features_blink[:,index_blink], features_idel[:,index_idel]), axis=1).T, np.concatenate((labels_blink[index_blink], labels_idel[index_idel]), axis=0)
train_data, train_labels = np.concatenate((np.delete(features_blink,index_blink,axis=1), np.delete(features_idel,index_idel,axis=1)), axis=1).T,np.concatenate((np.delete(labels_blink,index_blink), np.delete(labels_idel,index_idel)), axis=0)
print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)

# %%
from sklearn.svm import SVC
svc = SVC()
svc.fit(train_data, train_labels)
score = svc.score(test_data, test_labels)
print(score)
# %%
import pickle
# %%
with open('classifier.pkl', 'wb') as f:
    pickle.dump(svc, f)
# %%
print(train_data.shape)
# %%
