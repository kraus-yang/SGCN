import numpy as np
from sklearn.metrics import confusion_matrix    # 生成混淆矩阵函数
import matplotlib.pyplot as plt    # 绘图库


def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
    plt.title(title)    # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90,fontsize=10)    # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name,fontsize=10)    # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def get_confuse_matrix(wrong_file,right_file,label_name_file):
    f1 = open(wrong_file,'r')
    lines = f1.readlines()
    label = []
    pred = []
    for line in lines:
        line = line.strip('\n').split(',')[1:]
        label.append(int(line[0]))
        pred.append(int(line[1]))
    f1.close()
    f11 = open(right_file, 'r')
    lines = f11.readlines()
    for line in lines:
        line = line.strip('\n').split(',')
        label.append(int(line[0]))
        pred.append(int(line[1]))
    f2 = open(label_name_file,'r')
    name_list = f2.readlines()
    label_names = []
    for name in name_list:
        name = name[4:].strip()
        label_names.append(name)

    matrix = confusion_matrix(label, pred)
    np.savetxt("cm.txt",matrix)
    plot_confusion_matrix(matrix, label_names, "HAR Confusion Matrix")
    plt.savefig('./work_dir/cm.png', format='png',dpi=600)
    # plt.grid()
    plt.show()
    return matrix




















if __name__ == '__main__':
    num_class = 60
    wfile = './runs/ntu_cv_sgcn_test_joint_wrong.txt'
    rfile = './runs/ntu_cv_sgcn_test_joint_right.txt'
    name_file = './data/nturgbd_raw/label_name.txt'
    m = get_confuse_matrix(wfile,rfile,name_file)
    b=1