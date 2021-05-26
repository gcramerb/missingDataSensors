import os
import json
def classesNames(dataset):
    class_names = ""
    if dataset == 'MHEALTH.npz':
        actNameMHEALTH = {
            0: 'Standing still',
            1: 'Sittingand relaxing',
            2: 'Lying down',
            3: 'Walking',
            4: 'Climbing stairs',
            5: 'Waist bends forward',
            6: 'Frontal elevation of arms',
            7: 'Knees bending (crouching)',
            8: 'Cycling',
            9: 'Jogging',
            10: 'Running',
            11: 'Jump front and back'
        }
        class_names = actNameMHEALTH

    elif dataset == 'PAMAP2P.npz':

        actNamePAMAP2P = {
            0: 'lying',
            1: 'sitting',
            2: 'standing',
            3: 'ironing',
            4: 'vacuum cleaning',
            5: 'ascending stairs',
            6: 'descending stairs',
            7: 'walking',
            8: 'Nordic walking',
            9: 'cycling',
            10: 'running',
            11: 'rope jumping', }
        actNamePAMAP2P_v2 = {
            0: 'Lie',
            1: 'Sit',
            2: 'Stand',
            3: 'Iron',
            4: 'Break',
            5: 'Ascend stairs',
            6: 'Nordic walking',
            7: 'watching TV',
            8: 'computer work',
            9: 'car driving',
            10: 'ascending stairs',
            11: 'descending stairs',
            12: 'vacuum cleaning',
            13: 'ironing',
            14: 'folding laundry',
            15: 'house cleaning',
            16: 'playing soccer',
            17: 'rope jumping',
            18: 'other'}
        class_names = actNamePAMAP2P
    elif dataset == 'UTD-MHAD1_1s.npz':

        actNameUTDMHAD = {
            0: 'right arm swipe to the left',
            1: 'right arm swipe to the right',
            2: 'right hand wave',
            3: 'two hand front clap',
            4: 'right arm throw',
            5: 'cross arms in the chest',
            6: 'basketball shooting',
            7: 'draw x',
            8: 'draw circle clockwise',
            9: 'draw circle counter clockwise',
            10: 'draw triangle',
            11: 'bowling',
            12: 'front boxing',
            13: 'baseball swing from right',
            14: 'tennis forehand swing',
            15: 'arm curl',
            16: 'tennis serve',
            17: 'two hand push',
            18: 'knock on door',
            19: 'hand catch',
            20: 'pick up and throw'
        }
        class_names = actNameUTDMHAD
    elif dataset == 'UTD-MHAD2_1s.npz':

        actNameUTDMHAD2 = {
            0: 'jogging',
            1: 'walking',
            2: 'sit to stand',
            3: 'stand to sit',
            4: 'forward lunge',
            5: 'squat'}
        class_names = actNameUTDMHAD2

    elif dataset == 'WHARF.npz':

        actNameWHARF = {

            0: 'Standup chair',
            1: 'Comb hair',
            2: 'Sitdown chair',
            3: 'Walk',
            4: 'Pour water',
            5: 'Drink glass',
            6: 'Descend stairs',
            7: 'Climb stairs',
            8: 'Liedown bed',
            9: 'Getup bed',
            10: 'Use telephone',
            11: 'Brush teeth'}
        class_names = actNameWHARF

    elif dataset == 'USCHAD.npz':

        actNameUSCHAD = {
            0: 'Walking Forward',
            1: 'Walking Left',
            2: 'Walking Right',
            3: 'Walking Upstairs',
            4: 'Walking Downstairs',
            5: 'Running Forward',
            6: 'Jumping Up',
            7: 'Sitting',
            8: 'Standing',
            9: 'Sleeping',
            10: 'Elevator Up',
            11: 'Elevator Down'}
        class_names = actNameUSCHAD
    elif dataset == 'WISDM.npz':

        actNameWISDM = {
            0: 'Jogging',
            1: 'Walking',
            2: 'Upstairs',
            3: 'Downstairs',
            4: 'Sitting',
            5: 'Standing'
        }
        class_names = actNameWISDM
    return class_names
def saveAll(filename,result,config,curva,m):

    info = dict()
    info['resultado'] = result
    info['hiperparametros'] = config
    info['curva'] = curva
    info['missing'] = m

    path_save = filename + '/informations.json'
    with open(path_save, 'w') as fp:
        json.dump(info, fp)


def psnr_metric(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


def _psnr(visual_list, paths, opt, stats_file):
    rmse_per_activity = {}
    rmse_total = []

    split = opt.dataroot.split("/")
    dataset_name = split[-1] if split[-1] != "" else split[-2]

    # para cada arquivo de treino
    for i in range(len(visual_list)):
        sample_label = paths[i][0].split("_")[-1].split(".")[0]
        activity_label = target_names(dataset_name)[int(sample_label)]
        real = sensor_data(visual_list[i], "real_B", opt)
        fake = sensor_data(visual_list[i], "fake_B", opt)

        #psnr = peak_signal_noise_ratio(real, fake)
        psnr = psnr_metric(real, fake)

        if activity_label in rmse_per_activity:
            rmse_per_activity[activity_label].append(psnr)
        else:
            rmse_per_activity[activity_label] = [psnr]

        rmse_total.append(psnr)

    stats_file.write("PSNR metric\n")
    for k, v in rmse_per_activity.items():
        stats_file.write("{}: {}\n".format(k, np.mean(v)))
    stats_file.write("\nTOTAL: {}\n\n\n".format(np.mean(rmse_total)))

def _pearson_corr(visual_list, paths, opt, stats_file):
    axes_corr_per_activity = {}
    axes_corr_total = [[], [], []]

    split = opt.dataroot.split("/")
    dataset_name = split[-1] if split[-1] != "" else split[-2]

    # para cada arquivo de treino
    for i in range(len(visual_list)):
        sample_label = paths[i][0].split("_")[-1].split(".")[0]
        activity_label = target_names(dataset_name)[int(sample_label)]

        real_x, real_y, real_z = axes_data(visual_list[i], "real_B", opt)
        fake_x, fake_y, fake_z = axes_data(visual_list[i], "fake_B", opt)

        x_corr = pearsonr(real_x, fake_x)
        y_corr = pearsonr(real_y, fake_y)
        z_corr = pearsonr(real_z, fake_z)

        if activity_label in axes_corr_per_activity:
            axes_corr_per_activity[activity_label][0].append(x_corr)
            axes_corr_per_activity[activity_label][1].append(y_corr)
            axes_corr_per_activity[activity_label][2].append(z_corr)
        else:
            axes_corr_per_activity[activity_label] = [[x_corr], [y_corr], [z_corr]]

        axes_corr_total[0].append(x_corr)
        axes_corr_total[1].append(y_corr)
        axes_corr_total[2].append(z_corr)

    stats_file.write("Pearson Correlation Coefficient\n\n")

    for k, v in axes_corr_per_activity.items():
        stats_file.write("{}\n".format(k))
        stats_file.write("X: {}\n".format(np.mean(v[0])))
        stats_file.write("Y: {}\n".format(np.mean(v[1])))
        stats_file.write("Z: {}\n".format(np.mean(v[2])))

    stats_file.write("\nTOTAL\n")
    stats_file.write("X: {}\n".format(np.mean(axes_corr_total[0])))
    stats_file.write("Y: {}\n".format(np.mean(axes_corr_total[1])))
    stats_file.write("Z: {}\n".format(np.mean(axes_corr_total[2])))
