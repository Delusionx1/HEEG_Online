import os
import mne
import numpy as np
import random

def lodaDataFromFile(path):
    directory = os.fsencode(path)
    files = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".gdf"):
            file_path = directory.decode('ascii') +'//'+ filename
            files.append(file_path)
            continue
        else:
            continue
    print('End')
    for i in range(len(files)):
        print(files[i])
    raw=mne.io.read_raw_gdf(files[0],['EOG-left','EOG-central','EOG-right'])
    raw.rename_channels(mapping={'EEG-Fz':'Fz', 'EEG-0':'FC3', 'EEG-1':'FC1',\
     'EEG-2':'FCz', 'EEG-3':'FC2', 'EEG-4':'FC4', 'EEG-5':'C5', 'EEG-C3':'C3',\
     'EEG-6':'C1', 'EEG-Cz':'Cz', 'EEG-7':'C2', 'EEG-C4':'C4', 'EEG-8':'C6', \
     'EEG-9':'CP3', 'EEG-10':'CP1', 'EEG-11':'CPz', 'EEG-12':'CP2', 'EEG-13':'CP4',\
     'EEG-14':'P3', 'EEG-Pz':'Pz', 'EEG-15':'P4', 'EEG-16':'POz', 'EOG-left':'LPA',\
     'EOG-central':'Nz', 'EOG-right':'RPA'})

    montage = mne.channels.make_standard_montage('standard_1020', head_size=0.095)
    raw.set_montage(montage=montage)
    print(raw.ch_names)
    return raw
    
def dataSeparateByCategory(rawData):
    data = rawData.get_data()
    events, _ = mne.events_from_annotations(rawData)
    allLeftData = []
    allRightData = []
    allFootData = []
    allTongueData = []

    for i in range(events.shape[0]):
        if(events[i,2] == 7):
            leftData = data[:22,events[i,0]+250:events[i,0]+850]
            allLeftData.append(leftData)
        if(events[i,2] == 8):
            rightData = data[:22,events[i,0]+250:events[i,0]+850]
            allRightData.append(rightData)
        if(events[i,2] == 9):
            footData = data[:22,events[i,0]+250:events[i,0]+850]
            allFootData.append(footData)
        if(events[i,2] == 10):
            tongueData = data[:22,events[i,0]+250:events[i,0]+850]
            allTongueData.append(tongueData)
    
    allLeftData = np.array(allLeftData)
    allRightData = np.array(allRightData)
    allFootData = np.array(allFootData)
    allTongueData = np.array(allTongueData)
    print(allRightData.shape)
    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', \
                'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P3', 'Pz', 'P4', 'POz']
    info = mne.create_info(ch_names=ch_names,sfreq=250, ch_types='eeg')
    allLeftData = mne.EpochsArray(allLeftData,info=info)
    allRightData = mne.EpochsArray(allRightData,info=info)
    allFootData = mne.EpochsArray(allFootData,info=info)
    allTongueData = mne.EpochsArray(allTongueData,info=info)
    return allLeftData, allRightData, allFootData, allTongueData


def trainTestSplit(rawData, testRate):
    data = rawData.get_data()
    events, _ = mne.events_from_annotations(rawData)
    allLeftData = []
    allRightData = []
    allFootData = []
    allTongueData = []
    for i in range(events.shape[0]):
        if(events[i,2] == 7):
            leftData = data[:,events[i,0]+250:events[i,0]+850]
            allLeftData.append(leftData)
        if(events[i,2] == 8):
            rightData = data[:,events[i,0]+250:events[i,0]+850]
            allRightData.append(rightData)
        if(events[i,2] == 9):
            footData = data[:,events[i,0]+250:events[i,0]+850]
            allFootData.append(footData)
        if(events[i,2] == 10):
            tongueData = data[:,events[i,0]+250:events[i,0]+850]
            allTongueData.append(tongueData)
    print(len(allLeftData))
    print(len(allRightData))
    print(len(allFootData))
    print(len(allTongueData))
    allLeftData = np.array(allLeftData)
    allRightData = np.array(allRightData)
    allFootData = np.array(allFootData)
    allTongueData = np.array(allTongueData)
    allLeftData = allLeftData[:,:22,:]
    allRightData = allRightData[:,:22,:]
    allFootData = allFootData[:,:22,:]
    allTongueData = allTongueData[:,:22,:]
    print(allLeftData.shape)
    print(allRightData.shape)
    print(allFootData.shape)
    print(allTongueData.shape)
    random_list = random.sample(range(0, allLeftData.shape[0]), int(allLeftData.shape[0]*testRate))
    print(random_list)
    testLeftData = allLeftData[random_list,:,:]
    testRightData = allRightData[random_list,:,:]
    testFootData = allFootData[random_list,:,:]
    testTongueData = allTongueData[random_list,:,:]

    trainLeftData = np.delete(allLeftData,random_list,0)
    trainRightData = np.delete(allRightData,random_list,0)
    trainFootData = np.delete(allFootData,random_list,0)
    trainTongueData = np.delete(allTongueData,random_list,0)
    print(testFootData.shape, trainLeftData.shape)
    trainLeftDataTF = []
    trainRightDataTF = []
    trainFootDataTF = []
    trainTongueDataTF = []
    for i in range(trainLeftData.shape[0]):
        for j in range(5):
            trainLeftDataTF.append(standardlizeSig(trainLeftData[i,:,j*100:((j+2)*100)]))
            trainRightDataTF.append(standardlizeSig(trainRightData[i,:,j*100:(j+2)*100]))
            trainFootDataTF.append(standardlizeSig(trainFootData[i,:,j*100:(j+2)*100]))
            trainTongueDataTF.append(standardlizeSig(trainTongueData[i,:,j*100:(j+2)*100]))
    trainLeftDataTF = np.array(trainLeftDataTF)
    trainRightDataTF = np.array(trainRightDataTF)
    trainFootDataTF = np.array(trainFootDataTF)
    trainTongueDataTF = np.array(trainTongueDataTF)

    testLeftDataTF = []
    testRightDataTF = []
    testFootDataTF = []
    testTongueDataTF = []
    for i in range(testLeftData.shape[0]):
        for j in range(5):
            testLeftDataTF.append(standardlizeSig(testLeftData[i,:,j*100:((j+2)*100)]))
            testRightDataTF.append(standardlizeSig(testRightData[i,:,j*100:(j+2)*100]))
            testFootDataTF.append(standardlizeSig(testFootData[i,:,j*100:(j+2)*100]))
            testTongueDataTF.append(standardlizeSig(testTongueData[i,:,j*100:(j+2)*100]))
    testLeftDataTF = np.array(testLeftDataTF)
    testRightDataTF = np.array(testRightDataTF)
    testFootDataTF = np.array(testFootDataTF)
    testTongueDataTF = np.array(testTongueDataTF)
    print(testLeftDataTF.shape,testRightDataTF.shape,testFootDataTF.shape,testTongueDataTF.shape)
    print(trainLeftDataTF.shape,trainRightDataTF.shape,trainFootDataTF.shape,trainTongueDataTF.shape)
    trainleftLabels = [0] * trainLeftDataTF.shape[0]
    trainrightLabels = [1] * trainRightDataTF.shape[0]
    trainfootLabels = [2] * trainFootDataTF.shape[0]
    traintongueLabels = [3] * trainTongueDataTF.shape[0]
    allTrainLabels = trainleftLabels + trainrightLabels + trainfootLabels + traintongueLabels
    allTrainLabels = np.array(allTrainLabels)
    allTrainData = np.concatenate((trainLeftDataTF, trainRightDataTF, trainFootDataTF, trainTongueDataTF))

    testleftLabels = [0] * testLeftDataTF.shape[0]
    testrightLabels = [1] * testRightDataTF.shape[0]
    testfootLabels = [2] * testFootDataTF.shape[0]
    testtongueLabels = [3] * testTongueDataTF.shape[0]
    allTestLabels = testleftLabels + testrightLabels + testfootLabels + testtongueLabels
    allTestLabels = np.array(allTestLabels)
    allTestData = np.concatenate((testLeftDataTF, testRightDataTF, testFootDataTF, testTongueDataTF))
    print(allTestData.shape, allTrainData.shape)
    return allTrainData, allTrainLabels, allTestData, allTestLabels

def standardlizeSig(data):
    base = np.mean(data)
    std = np.std(data)
    standardlized_data = (data-base)/std
    del base, std
    return standardlized_data