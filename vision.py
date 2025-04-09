from matplotlib import pyplot as plt
def plot_history(history):
    epochs = range(1, len(history['train_acc']) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'][::len(history['train_loss'])//len(epochs)], label='Train Loss')
    plt.plot(epochs, history['val_loss'][::len(history['val_loss'])//len(epochs)], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], label='Train Acc')
    plt.plot(epochs, history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png')

def visualize_weights(model):
    """可视化网络权重"""
    W1 = model.params['W1'].reshape(32, 32, 3, -1)
    W1 = W1.transpose(3, 0, 1, 2)
    plt.figure(figsize=(10, 5))
    for i in range(20):
        plt.subplot(4, 5, i+1)
        img = (W1[i] - W1[i].min()) / (W1[i].max() - W1[i].min())
        plt.imshow(img)
        plt.axis('off')
    plt.suptitle('First Layer Filter Weights')
    
    
    plt.tight_layout()
    plt.savefig('Weights Visualize.png')