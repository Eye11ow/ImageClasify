import numpy as np
from data import load_and_preprocess_data
from model import ThreeLayerNN
from train import load_model
from vision import visualize_weights


def test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    test_acc = np.mean(y_pred == np.argmax(y_test, axis=1))
    print(f'Test accuracy: {test_acc:.4f}')
    return test_acc

if __name__ == "__main__":
    # 加载并预处理数据
    # X_train, y_train, X_val, y_val, X_test, y_test = load_and_preprocess_data()

     # 加载模型
    new_model = ThreeLayerNN(input_size=3072, hidden_size=1024, output_size=10)
    new_model = load_model(new_model,"best_model.pkl")
    visualize_weights(new_model)
    # 测试模型
    # test_acc = test(new_model, X_test, y_test)