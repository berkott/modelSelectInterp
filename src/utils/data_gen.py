from sklearn import linear_model
import torch


def get_data(alphas=[0.1, 0.5], N=21, d_d=5, train_samp_per_class=10000):
    data = {a: {b: {"y_test": 0, "y_hat": 0} for b in alphas} for a in alphas}

    train_samples_per_alpha = int(train_samp_per_class / len(alphas))

    for a in alphas:
        X_train = torch.randn(train_samples_per_alpha, N, d_d)
        X_test = torch.randn(train_samples_per_alpha, N, d_d)
        w = torch.randn(train_samples_per_alpha, d_d, 1)

        y_train = torch.squeeze(torch.matmul(X_train, w)) + torch.normal(0, a * 0.8, size=(train_samples_per_alpha, N))
        y_test = torch.squeeze(torch.matmul(X_test, w)) + torch.normal(0, a * 0.8, size=(train_samples_per_alpha, N))

        for b in alphas:
            results = torch.zeros((train_samples_per_alpha, N))
            for i in range(train_samples_per_alpha):
                reg = linear_model.Ridge(alpha=b)
                reg.fit(X_train[i], y_train[i])
                results[i] = torch.from_numpy(reg.predict(X_test[i]))
            data[a][b]["y_hat"] = results
            data[a][b]["y_test"] = y_test

    return data

def format_data(data_dict, train_samples_per_alpha=10000):
    alphas_merged = torch.cat([torch.ones(train_samples_per_alpha)*a for a in data_dict], dim=0)
    X_merged = torch.cat([torch.cat([torch.unsqueeze(data_dict[a][b]["y_hat"], 2) for b in data_dict[a]] + [torch.unsqueeze(data_dict[a][a]["y_test"], 2)], dim=2) for a in data_dict], dim=0)
    X_merged[:, -1, -1] = 0
    y_merged = torch.cat([data_dict[a][a]["y_test"][:, -1] for a in data_dict], dim=0)

    randperm = torch.randperm(alphas_merged.shape[0])

    alphas = alphas_merged[randperm]
    X = X_merged[randperm]
    y = y_merged[randperm]

    print(f"Alphas: {alphas.shape}, X: {X.shape}, y: {y.shape}")

    return alphas, X, y