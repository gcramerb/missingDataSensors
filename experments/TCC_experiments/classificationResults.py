from keras.layers import Input,Dense,Activation
from keras.models import Model
import custom_model as cm
from classifier import LatentHyperNet,custom_model
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.classification import accuracy_score, recall_score, f1_score
sys.path.insert(0, "C:\\Users\\gcram\\Documents\\GitHub\\TCC\\TCC\\")
from dataHandler import dataHandler
from utils import saveAll

if __name__ == '__main__':
    # Paper: Latent HyperNet: Exploring the Layers of Convolutional Neural Networks
    np.random.seed(12227)
    if (len(sys.argv) > 1):
        data_input_file = sys.argv[1]
        path = '/storage/datasets/HAR/LOSO/'
    else:
        data_input_file = 'C:\\Users\gcram\Documents\Smart Sense\Datasets\LOSO\\'
        path = data_input_file
    dataset_name ='USCHAD.npz'
    DH = dataHandler()
    DH.load_data(dataset_name=dataset_name, sensor_factor='1.0',path = path)
    n_class = len(pd.unique(DH.dataY))
    if dataset_name == 'UTD-MHAD1_1s' or dataset_name == 'UTD-MHAD2_1s':
        layers = [3, 6]
    else:
        layers = [3, 6, 9]
    avg_acc = []
    avg_recall = []
    avg_f1 = []
    missing_list = ['0.3', '0.4', '0.5', '0.6', '0.7']
    #impute_list = ['mean', 'defaut', 'last_value', 'interpolation', 'median','AE_mse', 'AE_sdtw']
    impute_list = ['mean', 'defaut', 'last_value', 'interpolation']
    Result = dict()
    n_folds = 14
    for miss in missing_list:
        for imp in impute_list:
            for fold_i in range(n_folds):
                dataset = dataset_name.split('.')[0]
                data_read = dataset + '_' + miss + '_' + imp + str(fold_i) + '.npz'
                path_read = path + data_read
                data_test = np.load(path_read, allow_pickle=True)
                test = data_test['deploy_data']
                test = np.expand_dims(test,axis = 1)
                Y = data_test['classes']
                
                DH.splitTrainTest(fold_i)
                train =  DH.dataXtrain[0]
                #train_2 = train
                train = np.expand_dims(train, axis=1)
                
            
                _, _, img_rows, img_cols = train.shape
                y_train = np.array(DH.dataYrawTrain)
                y_train_2 = y_train
                inp = Input((1, img_rows, img_cols))
                model = custom_model(inp, n_classes=n_class, dataset_name=dataset)
                model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adadelta')
                model.fit(train, y_train, batch_size=cm.bs, epochs=cm.n_ep,
                          verbose=0, callbacks=[cm.custom_stopping(value=cm.loss, verbose=2)],
                          validation_data=(train, y_train))
                hyper_net = LatentHyperNet(n_comp=19, model=model, layers=layers, dm_method='pls')
                hyper_net.fit(train,y_train)
                train = hyper_net.transform(train)
                test = hyper_net.transform(test)
                
                inp = Input((train.shape[1],))
                fc = Dense(n_class)(inp)
                model = Activation('softmax')(fc)
                model = Model(inp, model)
                model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='Adadelta')
                callbacks = [cm.custom_stopping(value=cm.loss, verbose=2)]

                model.fit(train, y_train, batch_size=len(train),
                          epochs=4*cm.n_ep,#The drawback of the method is that it requires more iterations to converge (loss <= cm.loss)
                           verbose=0, callbacks=callbacks, validation_data=(train, y_train))
        
        
        
                y_pred = model.predict(train_2)
                
                
                
                y_pred = np.argmax(y_pred, axis=1)
                y_true = y_train_2
                acc_fold = accuracy_score(y_true, y_pred)
                avg_acc.append(acc_fold)
                recall_fold = recall_score(y_true, y_pred, average='macro')
                avg_recall.append(recall_fold)
                f1_fold = f1_score(y_true, y_pred, average='macro')
                avg_f1.append(f1_fold)
                del model

            ic_acc = st.t.interval(0.9, len(avg_acc) - 1, loc=np.mean(avg_acc), scale=st.sem(avg_acc))
            ic_recall = st.t.interval(0.9, len(avg_recall) - 1, loc=np.mean(avg_recall), scale=st.sem(avg_recall))
            ic_f1 = st.t.interval(0.9, len(avg_f1) - 1, loc=np.mean(avg_f1), scale=st.sem(avg_f1))
            print('Mean Accuracy[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_acc), ic_acc[0], ic_acc[1]))
            print('Mean Recall[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_recall), ic_recall[0], ic_recall[1]))
            print('Mean F1[{:.4f}] IC [{:.4f}, {:.4f}]'.format(np.mean(avg_f1), ic_f1[0], ic_f1[1]))
            Result[data_read] = dict()
            Result[data_read]['acuracia'] = np.mean(avg_acc)
            Result[data_read]['reacall'] = np.mean(avg_recall)
            Result[data_read]['F1'] = np.mean(avg_f1)
            

    with open("metrics.json", "w") as write_file:
        json.dump(Result, write_file)