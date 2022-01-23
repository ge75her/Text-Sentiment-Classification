#load dataset
def load_training_data(path='training_label.txt'):
    # load train data
    # if data in train_nolabel.txt, don't need to return y
    if 'training_label' in path:
        with open(path, 'r',encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r',encoding='utf-8') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x

def load_testing_data(path='testing_data'):
    # test data
    with open(path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip() for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X

def evaluation(outputs, labels):
    #outputs => probability (float)
    #labels => labels
    outputs[outputs>=0.5] = 1 # prob>0.5 -> pos
    outputs[outputs<0.5] = 0 # prob<0.5 ->neg
    correct = torch.sum(torch.eq(outputs, labels)).item()
    return correct