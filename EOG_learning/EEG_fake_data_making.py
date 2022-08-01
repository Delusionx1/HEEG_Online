# %%
from DataLoadingAndPreProcessing import lodaDataFromFile,dataSeparateByCategory,standardlizeSig
import mne
import numpy as np
import pylab as plt
import random
#%%
dataPath = 'D:\MI Dataset\BCI2a\BCICIV_2a_gdf_T'
montage = mne.channels.make_standard_montage('standard_1020', head_size=0.095)
print(montage.dig)
rawData = lodaDataFromFile(dataPath)
allLeftData, allRightData, allFootData, allTongueData = dataSeparateByCategory(rawData)
allLeftData.set_montage(montage=montage)
allRightData.set_montage(montage=montage)
allFootData.set_montage(montage=montage)
allTongueData.set_montage(montage=montage)
#%%
allData = []
allData.append(allLeftData.get_data())
allData.append(allRightData.get_data())
allData.append(allFootData.get_data())
allData.append(allTongueData.get_data())

index_sample =random.sample(range(0,71), 20)

allLabels_train = []
allLabels_test = []
for i in range(72):
    if i not in index_sample:
        allLabels_train.append(0)
    else:
        allLabels_test.append(0)
for i in range(72):
    if i not in index_sample:
        allLabels_train.append(1)
    else:
        allLabels_test.append(1)
for i in range(72):
    if i not in index_sample:
        allLabels_train.append(2)
    else:
        allLabels_test.append(2)
for i in range(72):
    if i not in index_sample:
        allLabels_train.append(3)
    else:
        allLabels_test.append(3)

allLabels_train = np.array(allLabels_train)
allLabels_test = np.array(allLabels_test)

allData = np.array(allData)
print(allLabels_train.shape)
# %%
print(allLeftData.get_data().shape, allRightData.get_data().shape)
print(allData.shape)
print(index_sample)
allData_test = allData[:,index_sample,:,:]
allData_train = np.delete(allData, index_sample, axis=1)
# %%
starts = np.load(r'D:\Pytorch_learning\EOG_learning\EOG_starts.npy')
ends = np.load(r'D:\Pytorch_learning\EOG_learning\EOG_ends.npy')
eog_data = np.load(r'D:\Pytorch_learning\EOG_learning\EOG_online_fake_data_std.npy')
eog_data_real = np.load(r'D:\Pytorch_learning\EOG_learning\EOG_online_fake_data_real.npy')
errp_data = np.load(r'D:\Pytorch_learning\EOG_learning\Errp_data_std.npy')
errp_data_real = np.load(r'D:\Pytorch_learning\EOG_learning\Errp_data.npy')
errp_labels = np.load(r'D:\Pytorch_learning\EOG_learning\Errp_labels.npy')
print(errp_data.shape)
print(allData_test.shape)
print(errp_labels.shape)
#%%
test_labels = np.zeros((allData_test.shape[0],allData_test.shape[1],800))
print(ends.shape, allData_test.shape, test_labels.shape)
allData_test_std = np.zeros((allData_test.shape))
for i in range(allData_test.shape[0]):
    for j in range(allData_test.shape[1]):
        for k in range(allData_test.shape[2]):
            allData_test_std[i,j,k,:] = standardlizeSig(allData_test[i,j,k,:])
            # print( allData_test_std[i,j,k,:].shape)
            test_labels[i,j,:] = np.full(shape=(allData_test.shape[3],),fill_value = i)
#%%
generated_EEG_sig = np.zeros((22, 0))
generated_EEG_sig_real = np.zeros((22, 0))
generated_Errp_channels_real = np.zeros((errp_data_real.shape[1],0))
generated_labels_channle= np.zeros((1,0))
#%%
print(generated_Errp_channels_real.shape)
#%%
i = 0
mean = np.mean(allData_test_std[:, :, :, :600],axis=3)
mean_real = np.mean(allData_test[:, :, :, :600],axis=3)
mean_errp = np.mean(errp_data_real[:,:,:], axis = 2)
#%%
print(mean_errp.shape)
#%%
print(mean.shape)
while i < eog_data.shape[0]-20000:
    MI_type = random.randint(0, 3)
    trial_num = random.randint(0,19)
    trial_num_errp = random.randint(0,errp_data_real.shape[0]-1)
    if(i in ends):
        temp_data = allData_test_std[MI_type, trial_num, :, :600]
        temp_data_real = allData_test[MI_type, trial_num, :, :600]
        for ii in range(600):
            temp_errp = np.zeros((56,1))+ mean_errp[trial_num_errp,:].reshape(56,1)
            generated_Errp_channels_real = np.concatenate((generated_Errp_channels_real, temp_errp),axis=1)

        generated_EEG_sig = np.concatenate((generated_EEG_sig, temp_data), axis = 1)
        temp_labels = test_labels[MI_type, trial_num,:600]


        print(generated_labels_channle.shape, temp_labels.reshape(1,600).shape)

        generated_labels_channle = np.concatenate((generated_labels_channle,temp_labels.reshape(1,600)), axis=1)
        generated_EEG_sig_real = np.concatenate((generated_EEG_sig_real, temp_data_real), axis = 1)
        errp_index = random.randint(0,errp_data.shape[0]-1)
        errp_temp_data = errp_data[errp_index,:22,:]
        errp_temp_data_real = errp_data_real[errp_index,:22,:]
        errp_labels_temp = errp_labels[errp_index,:]

        generated_labels_channle = np.concatenate((generated_labels_channle,errp_labels_temp.reshape(1,260)), axis= 1)
        generated_EEG_sig = np.concatenate((generated_EEG_sig, errp_temp_data), axis = 1)
        generated_EEG_sig_real = np.concatenate((generated_EEG_sig_real, errp_temp_data_real), axis = 1)
        generated_Errp_channels_real = np.concatenate((generated_Errp_channels_real, errp_data_real[errp_index,:,:]), axis = 1)
        i = i + 860
    else:
        # print(generated_labels_channle.shape, np.array([0]).shape)
        generated_labels_channle = np.concatenate((generated_labels_channle,np.array([[0.5]])),axis = 1)
        temp_data = np.zeros((22,1))+ mean[MI_type, trial_num,:].reshape(22,1)
        temp_data_real = np.zeros((22,1))+ mean_real[MI_type, trial_num,:].reshape(22,1)
        generated_EEG_sig = np.concatenate((generated_EEG_sig,temp_data),axis=1)
        generated_EEG_sig_real = np.concatenate((generated_EEG_sig_real,temp_data_real),axis=1)
        temp_errp = np.zeros((56,1))+ mean_errp[trial_num_errp,:].reshape(56,1)
        generated_Errp_channels_real = np.concatenate((generated_Errp_channels_real, temp_errp), axis=1)
        i = i + 1
    print(i,'/', eog_data.shape[0])
print(generated_EEG_sig.shape)
print(generated_EEG_sig_real.shape)
print(generated_Errp_channels_real.shape)
print(generated_labels_channle.shape)
#%%
print(eog_data.shape, generated_EEG_sig.shape)
print(eog_data_real.shape, generated_EEG_sig_real.shape)
print(generated_labels_channle)
#%%
EEG_EOG_Cat = np.concatenate((eog_data[:37276].reshape(1,generated_EEG_sig.shape[1]),generated_labels_channle[:37276].reshape(1,generated_EEG_sig.shape[1])),axis=0)
EEG_EOG_Cat = np.concatenate((EEG_EOG_Cat,generated_EEG_sig),axis=0)
EEG_EOG_Cat_real = np.concatenate((eog_data_real[:37276].reshape(1,generated_EEG_sig_real.shape[1]),generated_EEG_sig_real),axis=0)
print(EEG_EOG_Cat.shape)

# %%
np.save('EEG_EOG_Cat.npy', np.array(EEG_EOG_Cat))
np.save('EEG_EOG_Cat_real.npy', np.array(EEG_EOG_Cat_real))
np.save('Errp_all_channels.npy', np.array(generated_Errp_channels_real))
#%%
print(EEG_EOG_Cat[0,:])
# %%
print(allRightData.get_data().shape)
print(np.max(allRightData.get_data()[1,:,:]), np.min(allRightData.get_data()[1,:,:]))
# %%
allData_train.shape
# %%
import scipy.signal as signal
from scipy.signal import cheb2ord
from FBCSP import FBCSP
data_for_train = []
data_for_test = []

train_labels = []
test_labels = []

for i in range(allData_train.shape[0]):
    for j in range(allData_train.shape[1]):
        for k in range(7):
            data_for_train.append(allData_train[i,j,:,k*50:400+k*50])
            train_labels.append(i)

for i in range(allData_test.shape[0]):
    for j in range(allData_test.shape[1]):
        for k in range(7):
            data_for_test.append(allData_test[i,j,:,k*50:400+k*50])
            test_labels.append(i)
data_for_train = np.array(data_for_train)
data_for_test = np.array(data_for_test)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
print(data_for_train.shape,allLabels_train.shape,data_for_test.shape)
print(allLabels_train)
# %%
class FilterBank:
    def __init__(self,fs):
        self.fs = fs
        self.f_trans = 2
        self.f_pass = np.arange(4,40,4)
        self.f_width = 4
        self.gpass = 3
        self.gstop = 30
        self.filter_coeff={}

    def get_filter_coeff(self):
        Nyquist_freq = self.fs/2

        for i, f_low_pass in enumerate(self.f_pass):
            f_pass = np.asarray([f_low_pass, f_low_pass+self.f_width])
            f_stop = np.asarray([f_pass[0]-self.f_trans, f_pass[1]+self.f_trans])
            wp = f_pass/Nyquist_freq
            ws = f_stop/Nyquist_freq
            order, wn = cheb2ord(wp, ws, self.gpass, self.gstop)
            b, a = signal.cheby2(order, self.gstop, ws, btype='bandpass')
            self.filter_coeff.update({i:{'b':b,'a':a}})

        return self.filter_coeff

    def filter_data(self,eeg_data):
        n_trials, n_channels, n_samples = eeg_data.shape
        filtered_data=np.zeros((len(self.filter_coeff),n_trials,n_channels,n_samples))
        for i, fb in self.filter_coeff.items():
            b = fb.get('b')
            a = fb.get('a')
            eeg_data_filtered = np.asarray([signal.lfilter(b,a,eeg_data[j,:,:]) for j in range(n_trials)])
            filtered_data[i,:,:,:]=eeg_data_filtered

        return filtered_data
# %%
freq = 250
fbank = FilterBank(freq)
fbank_coeff = fbank.get_filter_coeff()
filtered_data = fbank.filter_data(data_for_train)
m_filters = 2
y_classes_unique = np.unique(train_labels)
n_classes = len(np.unique(train_labels))
print(y_classes_unique, n_classes)
# %%
# index = np.arange(train_labels.shape[0])
# index_T = np.arange(test_labels.shape[0])
# print(index)
# np.random.shuffle(index)
# np.random.shuffle(index_T)
# print(filtered_data.shape)
#%%
fbcsp = FBCSP(m_filters)
fbcsp.fit(filtered_data,train_labels)

#%%
filtered_data_test = fbank.filter_data(data_for_test)
# %%

def get_multi_class_regressed(y_predicted):
    y_predict_multi = np.asarray([np.argmin(y_predicted[i,:]) for i in range(y_predicted.shape[0])])
    return y_predict_multi
    
from Classifier import Classifier
from sklearn.svm import SVR
y_train_predicted = np.zeros((train_labels.shape[0], n_classes), dtype=np.float)
y_test_predicted = np.zeros((test_labels.shape[0], n_classes), dtype=np.float)

print('---------------')
#%%
# shuffled_test = np.random.shuffle(np.arange(0,test_labels.shape[0]))
for j in range(n_classes):
        cls_of_interest = y_classes_unique[j]

        select_class_labels = lambda cls, y_labels: [0 if y == cls else 1 for y in y_labels]

        y_train_cls = np.asarray(select_class_labels(cls_of_interest, train_labels))
        y_test_cls = np.asarray(select_class_labels(cls_of_interest, test_labels))
        
        # print(y_train_cls.shape, y_test_cls.shape, filtered_data.shape)

        # x_features_train = fbcsp.transform(filtered_data,class_idx=cls_of_interest)
        # print(x_features_train.shape)
        # classifier_type = SVR(gamma='auto')
        # classifier = Classifier(classifier_type)
        # y_train_predicted[:,j] = classifier.fit(x_features_train,np.asarray(y_train_cls,dtype=np.float))
        
        x_features_train = fbcsp.transform(filtered_data,class_idx=cls_of_interest)
        x_features_test = fbcsp.transform(filtered_data_test,class_idx=cls_of_interest)

        classifier_type = SVR(gamma='auto')
        classifier = Classifier(classifier_type)
        y_train_predicted[:,j] = classifier.fit(x_features_train,np.asarray(y_train_cls,dtype=np.float))
        # print(y_train_cls)
        y_test_predicted[:,j] = classifier.predict(x_features_test)
#%%
# for j in range(n_classes):
#         x_features_test = fbcsp.transform(filtered_data_test,class_idx=cls_of_interest)
#         # y_test_predicted[:,j] = classifier.predict(x_features_test[0].reshape(1,-1))
#         print(x_features_test.shape)
#         y_test_predicted[:,j] = classifier.predict(x_features_test)

# print(y_train_predicted)
y_train_predicted_multi = get_multi_class_regressed(y_train_predicted)
# print(y_train_predicted_multi)
print(y_test_predicted.shape)
y_test_predicted_multi = get_multi_class_regressed(y_test_predicted)
tt_acc =np.sum(y_test_predicted_multi == test_labels, dtype=np.float) / len(test_labels)
tr_acc =np.sum(y_train_predicted_multi == train_labels, dtype=np.float) / len(train_labels)
print('----------------')
# print(y_test_predicted_multi, test_labels, train_labels)
print(tt_acc, tr_acc)
print(train_labels.shape, test_labels.shape,y_test_predicted_multi.shape)

# %%
import pickle
with open('fbcsp.pkl', 'wb') as f:
    pickle.dump(fbcsp, f)
with open('fbank.pkl', 'wb') as f:
    pickle.dump(fbank, f)
with open('fbcsp_classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)
# %%
with open('fbank.pkl', 'rb') as f:
    fbank_1 = pickle.load(f)
filtered_data = fbank_1.filter_data(data_for_train)
# %%
print(filtered_data.shape)
# %%
