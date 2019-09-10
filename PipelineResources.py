# import pandas as pd
# import io
# import os
# import time
# import pickle
import matplotlib.pyplot as plt
# import matplotlib.pylab
import numpy as np
from time import localtime, strftime
import seaborn as sns
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

chosen_features = ['open_change', 
                   'next_day_open_change', 
                   'open_change_wrt_close', 
                   'next_day_open_change_wrt_close', 
                   'high_change', 
                   'low_change',
                   'volume_change', 
                   'high_low_range', 
                   'close_change', 
                  'high_low_range_with_ref_close', 
                   'high_low_range_with_ref_open', 
                   'next_day_open_change_wrt_high',
                  'next_day_open_change_wrt_low']

chosen_label = 'gt_2.5'


def add_noise(data, numpy_data=False, noise_distribution=None, std=1.0):
    """
    This function adds noise to data array. Noise is added for some specific distribution, with a set
    standard deviation.
    :param data: numpy array
    :param numpy_data: bool, describing whether or not the data is a numpy array (otherwise it will be torch tensor)
    :param noise_distribution:
    :param std:
    :return: numpy array with added noise
    """
    numpy_tensor = False
    pytorch_tensor = False
    if not numpy_data:
        if 'numpy' in str(type(data)):
            numpy_tensor = True
            data = torch.tensor(data).float()
        else:
            pytorch_tensor = True

    if not noise_distribution:
        if numpy_data:
            noise_distribution = np.random.randn(data.shape[0], data.shape[1])
            data = data + noise_distribution
        else:
            noise_distribution = np.random.randn(data.shape[0], data.shape[1], data.shape[2])
            data = data + torch.tensor(noise_distribution).float().to(data.device)

    else:
        if numpy_data:
            data = data + noise_distribution(scale=std,
                                             size=[data.shape[0], data.shape[1]])
        else:
            data = data + torch.tensor(
                noise_distribution(scale=std, size=[data.shape[0], data.shape[1], data.shape[2]])).float().to(data.device)
    if pytorch_tensor:
        return data
    elif numpy_data:
        return data
    else:
        return data.numpy()


def augment_data(data, labels, numpy_data=False, number_of_items=None, random_pick=False):
    """
    Function to increase size of input array data with some random portion of the same array data.
    :param data:
    :param labels:
    :param numpy_data:
    :param number_of_items:
    :param random_pick:
    :return:
    """
    make_tensor = False

    if number_of_items is None:
        number_of_items = data.shape[0]
    if 'Tensor' in str(type(data)):
        data = data.numpy()
        make_tensor = True
    if numpy_data:
        new_data = np.zeros([data.shape[0] + number_of_items, data.shape[1]])
        new_labels = np.zeros(labels.shape[0] + number_of_items)
        new_data[:data.shape[0], :] = data
        new_labels[:labels.shape[0]] = labels
    else:
        new_data = np.zeros([data.shape[0] + number_of_items, data.shape[1], data.shape[2]])
        new_labels = np.zeros(labels.shape[0] + number_of_items)
        new_data[:data.shape[0], :, :] = data
        new_labels[:labels.shape[0]] = labels
    if random_pick:
        indexes = np.random.randint(0, data.shape[0], number_of_items)
    else:
        if number_of_items <= data.shape[0]:
            indexes = np.arange(0, number_of_items, 1)
        else:
            raise Exception("There was a problem with number_of_items")
    new_data[data.shape[0]:, :, :] = data[indexes]
    new_labels[labels.shape[0]:] = labels[indexes]
    if make_tensor:
        new_data = torch.tensor(new_data).float()
        new_labels = torch.tensor(new_labels).float()
    return new_data, new_labels
    
def compute_accuracy_score(net, x, y):
    """
    Compute accuracy score for Skorch module.
    :param net:
    :param x:
    :param y:
    :return:
    """
    #x = torch.tensor(val_dataset[:]['feature_value'])
    #y = torch.tensor(val_dataset[:]['labels'])
    if 'Tensor' in str(type(x)):
        pass
    else:
        x = torch.tensor(x)
        #y = torch.tensor(y)
    if 'Tensor' in str(type(y)):
        pass
    else:
        y = torch.tensor(y)
    y_pred = net.predict_proba(x)
    # bp()
    #labels = y.reshape(len(y_pred),1).long()
    labels = y.long()
    y_pred = torch.tensor(y_pred)
    _, predicted = torch.max(y_pred.data, 1)
    correct = (predicted == labels).sum().item()
    accuracy = correct / len(labels)
    return accuracy

# This function computes the accuracy on the test dataset
def compute_accuracy(net, testloader, device):
    """
    General accuracy computation function. Must pass pytorch tensor.
    :param net:
    :param testloader:
    :param device:
    :return:
    """
    # device = net.device

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in testloader:
            input_data = batch['feature_value']
            labels = batch['labels'].long()
            # bp()
            input_data, labels = input_data.to(device), labels.to(device)
            outputs = net(input_data)
            _, predicted = torch.max(outputs.data, 1)
            #predicted = torch.round(outputs).long()
            total += len(labels)
            correct += (predicted == labels).sum().item()
    return correct / total


def training_output(rs, train_dataset, test_dataset, val_dataset=None,
                    prefix_information='', save_results=False):
    """
    Function to help output the information of a trained model.
    :param rs:
    :param train_dataset:
    :param test_dataset:
    :param val_dataset:
    :param prefix_information:
    :param save_results:
    :return:
    """
    display_range = np.arange(0, 10, 1)
    y_pred = rs.predict_proba(train_dataset[:]['feature_value'])
    _, predicted = torch.max(torch.tensor(y_pred), 1)
    targets = train_dataset[:]['labels']
    print("TRAIN: predicted: {}, target: {}".format(predicted[display_range], targets[display_range]))
    plot_confusion_matrix(y_pred, targets, dataset_name='TRAIN',
                          prefix_information=prefix_information, save_results=save_results)

    if val_dataset is not None:
        y_pred = rs.predict_proba(val_dataset[:]['feature_value'])
        _, predicted = torch.max(torch.tensor(y_pred), 1)
        targets = val_dataset[:]['labels']
        print("VALIDATION: predicted: {}, target: {}".format(predicted[display_range], targets[display_range]))
        plot_confusion_matrix(y_pred, targets, dataset_name='VALIDATION',
                              prefix_information=prefix_information, save_results=save_results)

    y_pred = rs.predict_proba(test_dataset[:]['feature_value'])
    _, predicted = torch.max(torch.tensor(y_pred), 1)
    targets = test_dataset[:]['labels']
    print("TEST: predicted: {}, target: {}".format(predicted[display_range], targets[display_range]))
    plot_confusion_matrix(y_pred, targets, dataset_name='TEST',
                          prefix_information=prefix_information, save_results=save_results)




def plot_confusion_matrix(y_pred, y, prefix_information='',
                          dataset_name='', save_results=False,
                          y_pred_is_predicted_classes=False):
    """
    PLotting confusion matrix of different datasets (train, test, and validation).
    :param y_pred:
    :param y:
    :param prefix_information:
    :param dataset_name:
    :param save_results:
    :param y_pred_is_predicted_classes:
    :return:
    """
    if 'Tensor' in str(type(y)):
        y = y.numpy()
    if y_pred_is_predicted_classes:
        pass
    else:
        _, predicted = torch.max(torch.tensor(y_pred), 1)
        y_pred = predicted
    c_matrix = confusion_matrix(y, y_pred)
    accuracy = accuracy_score(y, y_pred)
    result_accuracy_string = "Accuracy of Net: {:.2f}".format(accuracy)
    print(result_accuracy_string)
    print("\nClassification report:")
    classfication_report_string = classification_report(y, y_pred)
    print(classfication_report_string)
    plt.figure()
    # plt.title('blah')
    fig, ax = plt.subplots(1, figsize=(4, 4))
    #ax.set_title(dataset_name + " Confusion matrix")
    ax.set_title(" Confusion matrix")
    sns.heatmap(c_matrix, cmap='Blues', annot=True, fmt='g', cbar=False)
    ax.set_xlabel('predictions')
    ax.set_ylabel('true labels')
    time_stamp_string = strftime("%Y_%m_%d_%H_%M_%S", localtime())
    # plt.show()
    # bp()
    if save_results:
        figure_filename = "{}{}_{}_acc-{}_{:s}.pdf".format(figures_folder, prefix_information, dataset_name,
                                                           str(accuracy), time_stamp_string)
        print(figure_filename)
        plt.savefig(temp_test_results_folder + figure_filename, format='pdf')
        mod_classfication_report_string = classfication_report_string[13:]
        html_results_content = """
            <html>
                <body>
                <p>
                    {}
                </p>
                    Classification Report: <br>
                    <pre>
                        {}
                    </pre>
                    <br>
                     <figure>
                      <img src="{}" alt="Classification Report" width="272" height="278">
                    </figure> 
                </body>

            </html>
        """.format(result_accuracy_string, mod_classfication_report_string, figure_filename)
        html_result_file_name = "{}{}_{}_acc-{}_{:s}.html".format(temp_test_results_folder, prefix_information,
                                                                  dataset_name, str(accuracy), time_stamp_string)
        html_result_file = open(file=html_result_file_name, mode='w')
        html_result_file.write(html_results_content)
        html_result_file.close()

    return accuracy


def get_input_sets(dataset_df, chosen_features, chosen_label, x_test=True,
                   random_state=1, test_pct=0.25, validation=0.25,
                   number_of_days_per_sample=1):
    """
    Setup dataset for testing. Tthe output dataframe is specific for a set of chosen features.
    :param dataset_df:
    :param chosen_features:
    :param chosen_label:
    :param x_test:
    :param random_state:
    :param test_pct:
    :param validation:
    :param number_of_days_per_sample:
    :return:
    """
    x_train = dataset_df[chosen_features].values
    x_train = x_train[
              1:x_train.shape[0] - 3]  # Adjusting for zeros in the beginning and at the end of some of the features
    y_train = dataset_df[chosen_label].astype(int).values[
              2:dataset_df.index.size - 2]  # adjusting the labels to be in the future (one day ahead)
    print("number of days per samples: {}".format(number_of_days_per_sample))
    if number_of_days_per_sample > 1:
        x_train_new = np.zeros(
            (x_train.shape[0] - number_of_days_per_sample, x_train.shape[1] * number_of_days_per_sample),
            dtype=np.float32)
        days_index = np.arange(number_of_days_per_sample - 1, stop=-1, step=-1)
        new_index = 0

        for index in range(number_of_days_per_sample - 1, x_train.shape[0] - 1):
            # x_index = np.copy(index)
            column_index = 0
            for day_index in days_index:
                x_train_new[new_index,
                column_index * x_train.shape[1]: (column_index + 1) * x_train.shape[1]] = x_train[index, :]
                index -= 1
                column_index += 1
            new_index += 1
        x_train = x_train_new
        y_train = y_train[number_of_days_per_sample:]

    print(y_train.shape)
    print(dataset_df[chosen_label].astype(int).values[:20])
    print(y_train[:20])
    print(x_train.shape)
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      test_size=validation,
                                                      random_state=random_state)
    if x_test:
        x_train, x_test, y_train, y_test = train_test_split(x_train,
                                                            y_train,
                                                            test_size=test_pct,
                                                            random_state=random_state)
    else:
        x_test = None
        y_test = None

    return x_train, x_val, x_test, y_train, y_val, y_test
    

def get_fft(data, absolute_value=False):
    """
    Compute the fft for a specific input dataset.
    :param data:
    :param absolute_value:
    :return:
    """
    fft_data = np.fft.fft(data, axis=0) * 1/len(data)
    if not absolute_value:
        return fft_data
    else:
        abs_fft_data = abs(fft_data[:int(len(fft_data) / 2)])
        return abs_fft_data


def modify_dataset_channels(x_train, x_val, x_test=False, compute_fft=False,
                            multi_channel=False):
    """
    Function to modify the input channels of the data. In essence, we add a channel, creating a tensor of NxDxF,
    where N = number of observations, D = number of dimensions, and F = number of features.
    :param x_train:
    :param x_val:
    :param x_test:
    :param compute_fft:
    :param multi_channel:
    :return:
    """
    if multi_channel:

        if compute_fft:
            z_train = []
            z_val = []
            for row_index in range(x_train.shape[0]):
                z_train.append(get_fft(x_train[row_index], absolute_value=True))
            for row_index in range(x_val.shape[0]):
                z_val.append(get_fft(x_val[row_index], absolute_value=True))
            z_train = np.array(z_train, dtype=np.float)
            z_val = np.array(z_val, dtype=np.float)
            x_train_new = np.zeros((x_train.shape[0], 2, x_train.shape[1]), dtype=np.float32)
            x_val_new = np.zeros((x_val.shape[0], 2, x_val.shape[1]), dtype=np.float32)

            x_train_new[:, 0, :] += x_train
            z_new = np.concatenate((z_train, np.zeros((z_train.shape[0], x_train.shape[1] - z_train.shape[1]))), axis=1)
            x_train_new[:, 1, :] += z_new

            print(x_train_new.shape)

            x_val_new[:, 0, :] += x_val
            z_new = np.concatenate((z_val, np.zeros((z_val.shape[0], x_val.shape[1] - z_val.shape[1]))), axis=1)
            x_val_new[:, 1, :] += z_new

            print(x_val_new.shape)

            if x_test is not False:
                z_test = []
                for row_index in range(x_test.shape[0]):
                    z_test.append(get_fft(x_test[row_index], absolute_value=True))
                z_test = np.array(z_test, dtype=np.float32)
                x_test_new = np.zeros((x_test.shape[0], 2, x_test.shape[1]), dtype=np.float32)
                x_train_new[:, 0, :] += x_train
                z_new = np.concatenate((z_test, np.zeros((z_test.shape[0], x_test.shape[1] - z_test.shape[1]))), axis=1)
                x_test_new[:, 1, :] += z_new
                x_test = x_test_new

            x_train = x_train_new
            x_val = x_val_new
        else:
            raise Exception(NotImplementedError)

    else:
        if compute_fft:
            z_train = []
            z_val = []
            for row_index in range(x_train.shape[0]):
                z_train.append(get_fft(x_train[row_index], absolute_value=True))
            for row_index in range(x_val.shape[0]):
                z_val.append(get_fft(x_val[row_index], absolute_value=True))
            z_train = np.array(z_train, dtype=np.float)
            z_val = np.array(z_val, dtype=np.float)
            x_train_new = np.zeros((x_train.shape[0], x_train.shape[1] + z_train.shape[1]), dtype=np.float32)
            x_val_new = np.zeros((x_val.shape[0], x_val.shape[1] + z_val.shape[1]), dtype=np.float32)
            x_train_new[:, :x_train.shape[1]] += x_train
            x_train_new[:, x_train.shape[1]:] += z_train
            x_val_new[:, :x_val.shape[1]] += x_val
            x_val_new[:, x_val.shape[1]:] += z_val
            x_train = x_train_new
            x_val = x_val_new
            x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
            x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
            print(x_train.shape)
            if x_test is not False:
                raise Exception(NotImplementedError)
        else:
            x_train = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
            x_val = x_val.reshape(x_val.shape[0], 1, x_val.shape[1])
            if x_test is not False:
                x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
            print(x_train.shape)
            print(x_val.shape)

    return x_train, x_val, x_test

def decrease_class_rows(dataset, labels, class_number=0,
                        percentage_cut=0.50):
    """
    Function to help balance the observations used. This only works for a binary dataset.
    :param dataset:
    :param labels:
    :param class_number:
    :param percentage_cut:
    :return:
    """
    sample_rows = np.where(labels == class_number)[0]
    number_of_zeros = len(sample_rows)
    print(number_of_zeros)
    new_number_of_zeros = int(percentage_cut * number_of_zeros)
    print(new_number_of_zeros)
    np.random.shuffle(sample_rows)
    new_sample_rows = sample_rows[:new_number_of_zeros]
    a = set(np.arange(dataset.shape[0]))
    b = set(new_sample_rows)
    c = a - b
    c = list(c)
    dataset_new = dataset[c]
    labels_new = labels[c]
    dataset = dataset_new
    labels = labels_new
    print(labels.shape)
    print(dataset.shape)
    return dataset, labels

def double_class_rows(dataset, labels, class_number=1, random_state=1):
    """
    This function helps balance by adding (or cloning) observations of a specific type.
    :param dataset:
    :param labels:
    :param class_number:
    :param random_state:
    :return:
    """
    sample_rows = np.where(labels == class_number)
    dataset_new = dataset[sample_rows]
    dataset = np.concatenate((dataset, dataset_new))
    labels = np.concatenate((labels, labels[sample_rows]))
    dataset, x_test, labels, y_test = train_test_split(dataset,
                                                      labels,
                                                      test_size=0,
                                                      random_state=random_state)
    print(dataset.shape)
    return dataset, labels