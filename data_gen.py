from sklearn import linear_model
import torch

def get_data(alphas=[0.1, 0.5], N=21, d_d=5, train_samp_per_class=10000):
    data = {a: {b: {"y_hat": 0, "y_test": 0} for b in alphas} for a in alphas}

    for a in alphas:
        x_train = torch.normal(0, 1, size=(train_samp_per_class, N, d_d))
        x_test = torch.normal(0, 1, size=(train_samp_per_class, N, d_d))

        w = torch.normal(0, 1, size=(train_samp_per_class, d_d, 1))

        y_train = torch.squeeze(torch.matmul(x_train, w)) + torch.normal(0, a*1.5, size=(train_samp_per_class, N))
        y_test = torch.squeeze(torch.matmul(x_test, w)) + torch.normal(0, a*1.5, size=(train_samp_per_class, N))

        for b in alphas:
            results = torch.zeros((train_samp_per_class, N))
            for i in range(train_samp_per_class):
                reg = linear_model.Ridge(alpha=b)
                reg.fit(x_train[i], y_train[i])
                results[i] = torch.from_numpy(reg.predict(x_test[i]))
            data[a][b]["y_hat"] = results
            data[a][b]["y_test"] = y_test

    return data