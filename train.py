import numpy as np
import pickle
from data import load_and_preprocess_data
from model import ThreeLayerNN
from vision import plot_history,visualize_weights


def sgd_update(params, grads, learning_rate):
    for key in params:
        params[key] -= learning_rate * grads[key]
    return params

def train(model, X_train, y_train, X_val, y_val, num_epochs=20, batch_size=128, learning_rate=1e-3, lr_decay=0.95):
    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [],'val_acc': []}
    best_val_acc = 0.0
    best_params = {}

    for epoch in range(num_epochs):
        indices = np.arange(num_train)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]

        for i in range(iterations_per_epoch):
            start = i * batch_size
            end = start + batch_size
            X_batch = X_train[start:end]
            y_batch = y_train[start:end]
            
            train_loss = model.loss(X_batch, y_batch)
            grads = model.backward(X_batch, y_batch)
            history['train_loss'].append(train_loss)
            model.params = sgd_update(model.params, grads, learning_rate)
        
        train_pred = model.predict(X_train)
        train_acc = np.mean(train_pred == np.argmax(y_train, axis=1))
        val_pred = model.predict(X_val)
        val_loss = model.loss(X_val, y_val)
        val_acc = np.mean(val_pred == np.argmax(y_val, axis=1))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f'Epoch {epoch+1}/{num_epochs}: train_loss = {train_loss:.4f}, train_acc = {train_acc:.4f}, val_loss = {val_loss:.4f},val_acc = {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}

        learning_rate *= lr_decay

    model.params = best_params
    return history, model


def save_model(model, filename='best_model.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(model.params, f)
    print('Model saved.')

def load_model(model, filename='best_model.pkl'):
    with open(filename, 'rb') as f:
        model.params = pickle.load(f)
    print('Model loaded.')
    return model

if __name__ == '__main__':
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    
    hidden_size = 1024
    reg = 1e-4
    lr = 0.05
    # 初始化模型
    model = ThreeLayerNN(input_size=3072, hidden_size=hidden_size, output_size=10, reg=reg)
    
    # 训练模型
    history, trained_model = train(model, X_train, y_train, X_val, y_val, num_epochs=100, batch_size=128, learning_rate=lr, lr_decay=0.95)
    
    # 可视化
    plot_history(history)
    visualize_weights(trained_model)

    
    # 保存模型
    save_model(trained_model)
    
   