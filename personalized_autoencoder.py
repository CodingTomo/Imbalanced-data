# torch
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# utils
import utilities
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix


class AutoEncoder(nn.Module):
    def __init__(self, inp_size, hidden_dim, out_dim):
        super().__init__()

        current_dim = inp_size

        self.autoencoder = nn.Sequential()

        i = 0

        for dim in hidden_dim:
            self.autoencoder.add_module(name='Linear_module_' + str(i), module=nn.Linear(current_dim, dim, bias=True))
            self.autoencoder.add_module(name='ReLU_module_' + str(i), module=nn.LeakyReLU(negative_slope=0.4))
            self.autoencoder.add_module(name='Dropout_module' + str(i), module=nn.Dropout(p=0.2))
            current_dim = dim
            i = i + 1

        self.autoencoder.add_module(name='Linear_module_out', module=nn.Linear(current_dim, out_dim))
        self.autoencoder.add_module(name='module_LReLU_out', module=nn.Sigmoid())

        init_weights(self)

    def forward(self, x):
        return self.autoencoder(x)


def init_weights(my_module):
    for sub_module in my_module.modules():
        if isinstance(sub_module, nn.Linear):
            nn.init.kaiming_normal_(sub_module.weight, nonlinearity='leaky_relu')
            nn.init.constant_(sub_module.bias, 0)


def batch_preparation(data, label, batch_size):
    X = data[data.select_dtypes('number').columns.tolist()]
    y = data[label]
    X = X.drop([label], axis=1)

    sc = MinMaxScaler()
    X = pd.DataFrame(data=sc.fit_transform(X), columns=X.columns)

    X_torch = torch.from_numpy(X.values).float()

    data_loader = DataLoader(X_torch, batch_size=batch_size)

    return X_torch, y, data_loader


def train_ae(X_torch, data_loader, model, n_epochs, tb, lr):
    log_interval = 10000

    loss_function = nn.L1Loss(reduction='mean')
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    data = torch.autograd.Variable(X_torch)

    reconstruction_loss_all = 0

    model.train()

    for epoch in range(n_epochs):
        i = 1
        loss_on_epoch = 0
        for mini_batch in data_loader:
            optim.zero_grad()
            batch_recon = model(mini_batch.float())
            loss = loss_function(batch_recon, mini_batch)

            loss_on_epoch += loss.item()

            loss.backward()
            optim.step()

            if (i + 1) % log_interval == 0:
                print("Epoch {} - batch {}. \n The error in this mini_bach was: {:.3f}".format(epoch + 1,
                                                                                               i + 1,
                                                                                               loss.item()))

            i += 1
        print("Epoch {}. \n The mean error on this epoch was: {:.3f}".format(epoch + 1,
                                                                             loss_on_epoch / len(data_loader)))

        print("... {:.0%} of train completed".format((epoch + 1) / n_epochs))

        tb.add_scalar('Loss', loss_on_epoch, epoch+1)

    model.eval()
    reconstruction = model(data)
    test_loss = nn.L1Loss(reduction='none')
    reconstruction_loss_all = test_loss(reconstruction, data)
    model.train()

    return reconstruction_loss_all


def prediction_quality(reconstruction_loss, y, params, tb):
    reconstruction_loss_np = reconstruction_loss.detach().numpy().sum(axis=1)

    # utilities.plot_outlier_scores(y, reconstruction_loss_np, bw=0.1,
    #                              title='Fraud, Personalized Autoencoder. (epochs={})'.format(params.torch_epochs))

    contaminated_record = int(reconstruction_loss_np.shape[0] * 0.02)

    loss_df = pd.DataFrame(reconstruction_loss_np, columns=['loss'])
    threshold = loss_df.sort_values(ascending=False, by='loss').head(contaminated_record).tail(1).values
    loss_df['label'] = loss_df['loss'].map(lambda x: 1 if x >= threshold[0] else 0)

    report = pd.concat([pd.Series(y, name='true', dtype='int32'),
                        pd.Series(loss_df['label'], name='label', dtype='int32')],
                       axis=1)

    report['true positive'] = np.where((report['true'] == 1) & (report['label'] == 1), 1, 0)
    report['false negative'] = np.where((report['true'] == 1) & (report['label'] == 0), 1, 0)
    report['false positive'] = np.where((report['true'] == 0) & (report['label'] == 1), 1, 0)
    report['true negative'] = np.where((report['true'] == 0) & (report['label'] == 0), 1, 0)

    coverage = report['true positive'].sum() / report['true'].sum()

    print('Checking {} case, the coverage on frauds is about {:.0%}'
          .format(report['label'].sum(),
                  coverage))

    tb.add_text(tag='Report', text_string='Checking {} case, the coverage on frauds is about {:.0%}'
                .format(report['label'].sum(),
                        coverage))

    cm = confusion_matrix(y_true=report['true'].values, y_pred=report['label'].values)

    tb.add_text(tag='Report', text_string='True Negative: {};\n True Positive: {};'
                                          '\n False Positive {};\n False Negative {}.'
                .format(cm[0][0], cm[1][1], cm[0][1], cm[1][0]), global_step=None, walltime=None)

    tb.add_hparams(hparam_dict={'learning rate': params.torch_lr,
                                'batch size': params.torch_batch,
                                'hidden_dimensions': str(params.torch_hidden_dim),
                                'epochs': params.torch_epochs,
                                'loss': 'L1',
                                'activation output': 'sigmoid',
                                'dropout': 'active'
                                }, metric_dict={})

    tb.close()

    return report


def personalized_autoencoder(data, params):
    comment = "Run's ID = {}".format(params.torch_id)
    tb = SummaryWriter(comment=comment)

    X_torch, y,  data_loader = batch_preparation(data, params.label, params.torch_batch)
    model = AutoEncoder(params.torch_input_dim,
                        params.torch_hidden_dim,
                        params.torch_output_dim).float()

    tb.add_graph(model=model, input_to_model=next(iter(data_loader)))

    reconstruction_loss = train_ae(X_torch, data_loader, model, params.torch_epochs, tb, params.torch_lr)
    report = prediction_quality(reconstruction_loss, y, params, tb)

    return report
