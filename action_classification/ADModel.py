import random
import torch
from torch import nn
from .tools import save_pkl,read_pkl

def data_iter(train_data,label_data,batch_size=5,is_shuffle=True):
        num_example = len(train_data)
        if is_shuffle:
            indices =list(range(num_example))
            random.shuffle(indices)
            for i in range(0,num_example,batch_size):
                batch_indices = torch.tensor(indices[i:min(i+batch_size,num_example)])
                yield train_data[batch_indices],label_data[batch_indices]
        else:
            indices = list(range(num_example))
            for i in range(0,num_example,batch_size):
                batch_indices = torch.tensor(indices[i:min(i + batch_size, num_example)])
                yield train_data[batch_indices],label_data[batch_indices]

def accuracy(y_hat,y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1) #按行 获取最大值的下标
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator:  #@save
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def evaluate_accuracy(net, data_iter):  #@save
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class ADModel(nn.Module):
    def __init__(self):
        super(ADModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (1, 3)),
            nn.Sigmoid(),
            nn.Conv2d(32, 16, (1, 5)),
            nn.Sigmoid(),
            nn.Conv2d(16, 8, (1, 3)),
            nn.ReLU()
        )
        self.classification =nn.Sequential(
            nn.Linear(264,32),
            nn.Linear(32,4),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.model(x)
        x = torch.flatten(x,start_dim=1)
        out = self.classification(x)
        return out

if __name__=="__main__":

    # from torchsummary import summary
    # model = ADModel()
    # summary(model,(1,11,11),batch_size=2,device="cpu")

    leaning_rate = 0.05
    batch_size = 5
    device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ADModel()
    Loss = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(),leaning_rate)
    if torch.cuda.is_available():
        Loss.cuda()
        Loss.to(device)
        model.cuda()
        model.to(device)
    model.train()
    num_epoch = 30
    total_train_step = 0
    data = read_pkl("./train_v5.pkl")
    train_data = torch.tensor(data["keypoints"],dtype=torch.float32)
    train_label = torch.tensor(data["labels"])
    for epoch in range(num_epoch):
        print("-------------training data {} epoch---------------".format(epoch+1))
        for train_d,train_l in data_iter(train_data,train_label,batch_size=batch_size):
            if torch.cuda.is_available():
                train_d = train_d.cuda()
                train_l = train_l.cuda()
            out = model(train_d)
            loss = Loss(out,train_l.long())
            optim.zero_grad()
            loss.backward()
            optim.step()
            total_train_step += 1
            # if total_train_step % 20 == 0:
            #     print("-------训练次数{}, Loss:{}".format(total_train_step, loss))
        test_acc = evaluate_accuracy(model,data_iter(train_data,train_label,batch_size=30))
        print("--------------test acc : {} -----------".format(test_acc))
    torch.save(model, "model_latest_test_acc_{}_v2.pth".format(test_acc))
