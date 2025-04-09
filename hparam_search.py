from data import load_and_preprocess_data
from model import ThreeLayerNN
from train import train



def hyperparameter_search(X_train, y_train, X_val, y_val, param_grid):
    results = {}
    best_val_acc = 0.0
    best_params = None
    for lr in param_grid['learning_rate']:
        for hidden_size in param_grid['hidden_size']:
            for reg in param_grid['reg_strength']:
                print(f"Training with lr={lr}, hidden_size={hidden_size}, reg={reg}")
                model = ThreeLayerNN(input_size=3072, hidden_size=hidden_size, output_size=10, reg=reg)
                history, trained_model = train(model, X_train, y_train, X_val, y_val, num_epochs=20, batch_size=128, learning_rate=lr)
                val_acc = history['val_acc'][-1]
                results[(lr, hidden_size, reg)] = val_acc
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_params = (lr, hidden_size, reg)
    print('Best hyperparameters:', best_params, 'with validation accuracy:', best_val_acc)
    return results, best_params
    
    

if __name__ == "__main__":
    # 加载并预处理数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()
    
    # 超参数搜索
    param_grid = {
        'learning_rate': [0.05,0.01,0.005],
        'hidden_size': [256, 512, 1024],
        'reg_strength': [1e-3, 1e-4, 1e-5]
    }
    results, best_params = hyperparameter_search(X_train, y_train, X_val, y_val, param_grid)