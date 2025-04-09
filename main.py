import argparse
from model import ThreeLayerNN
from train import load_model, train,save_model
from test import test
from hparam_search import hyperparameter_search
from data import load_and_preprocess_data
from vision import plot_history, visualize_weights

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Program with train/test/search modes')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'test', 'search'],
                      help='Mode of operation: train, test, or search')
    parser.add_argument('--hidden_size', type=int, default=1024,
                      help='Hidden size for the model')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                      help='Learning rate for training')
    parser.add_argument('--reg', type=float, default=0.0001,
                      help='Regularization strength')
    parser.add_argument('--model_path', type=str, default='best_model.pkl',
                      help='Path to save/load the model')
    args = parser.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()

    if args.mode == 'train':
        # 初始化模型
        model = ThreeLayerNN(input_size=3072, hidden_size=args.hidden_size, output_size=10, reg=args.reg)
        
        # 训练模型
        history, trained_model = train(model, X_train, y_train, X_val, y_val, num_epochs=100, batch_size=128, learning_rate=args.learning_rate, lr_decay=0.95)
        
        # 可视化训练过程
        plot_history(history)
        visualize_weights(trained_model)
        # 保存模型
        save_model(trained_model)

    elif args.mode == 'test':
         # 加载模型
        new_model = ThreeLayerNN(input_size=3072, hidden_size=args.hidden_size, output_size=10)
        new_model = load_model(new_model,args.model_path)
    
        # 测试模型
        test_acc = test(new_model, X_test, y_test)

    elif args.mode == 'search':
        # 超参数搜索
        param_grid = {
            'learning_rate': [0.05,0.01,0.005],
            'hidden_size': [256, 512, 1024],
            'reg_strength': [1e-3, 1e-4, 1e-5]
        }
        results, best_params = hyperparameter_search(X_train, y_train, X_val, y_val, param_grid)
        print("\n提示：可以使用以下参数进行完整训练：")
        print(f"(lr, hidden_size, reg): {best_params}")