import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# Utils

def save_training_checkpoint(torch_model, isBest, episode_count):
    """Save trained models
    """
    filename = 'model_' + str(episode_count) + '.ckpt'
    if isBest:
        filename = 'model_best.ckpt'
    torch.save(torch_model, filename)

def load_checkpoint(target_model, filename):
    """load trained models ==> target_model
    """
    target_model.load_state_dict(torch.load(filename))
    print('Models from {} loaded sucessfully'.format(filename))


def weight_init(m):
    """Initialize the network
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


def plot(y_score, y_true, path):
    """plot the ROC of each label
    """
    fpr_list, tpr_list, auc_list = [], [], []
    label_ct = 0
    # single label
    for y_t, y_s in zip(y_true.numpy().transpose(), y_score.numpy().transpose()):
        fpr, tpr, threshold = metrics.roc_curve(y_t, y_s)
        auc = metrics.roc_auc_score(y_t, y_s, average='macro')
        auc_list.append(auc)
        fpr_list.append(np.array(fpr))
        tpr_list.append(np.array(tpr))
        label_ct += 1
    
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    for i in range(label_ct):
        plt.plot(fpr_list[i], tpr_list[i],
                lw=1.5, label='Label[%d] ROC-curve (area = %0.2f)' % (i, auc_list[i])) 
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Evaluation on dev set for protein photography classification')
    plt.legend(loc="lower right")
    plt.savefig(path + 'roc-curve.png')
    plt.close()