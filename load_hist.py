import os
import pickle
import matplotlib.pyplot as plt


hist = './hist'  # 损失、准确率保存路径
log_file = 'model_alexnet_20240530-095321.pkl'  # alexnet
# log_file = 'model_alexnet_20240529-220813.pkl'  # alexnet_latent
# log_file = 'model_resnet_20240529-220809.pkl'  # resnet
with open(os.path.join(hist, log_file), 'rb') as f:
    logs = pickle.load(f)
    train_loss = logs['train_loss']
    train_acc = logs['train_acc']
    time_cost = logs['time_cost']
    plt.figure()
    plt.plot(train_loss, label='train loss')
    plt.plot(train_acc, label='train acc')
    plt.axhline(y=1.00, color='grey', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.savefig(f"./plots/train_{log_file}.png", bbox_inches='tight')
    print(f"Epochs: {len(train_loss)}")
    print("Train Losses:", train_loss)
    print("Train Accuracies:", train_acc)
    print('Training complete in {:.0f}m {:.0f}s'.format(time_cost // 60, time_cost % 60))
