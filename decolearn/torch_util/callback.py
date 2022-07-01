"""
Idea basically come from tf.Keras.Callback and redefined by W.G.  - 2020.02.14

Callbacks: utilities called at certain points during model training.
"""

import torchvision

import logging
import pprint
import shutil
import warnings

from torch_util.common import *


def pformat_module(module: torch.nn.Module):
    structure = ''
    for name, p in module.named_parameters():
        if p.requires_grad:
            structure += str(name) + ': ' + str(type(p.data)) + " " + str(p.size()) + "\n"

    num_params = format(sum(p.numel() for p in module.parameters() if p.requires_grad), ',')

    return structure, num_params


def normalize_(x: torch.Tensor):
    try:
        x = x - torch.min(x)
        x = x / torch.max(x)
    except:
        pass

    try:
        x = x - np.amin(x)
        x = x / np.amax(x)
    except:
        pass

    return x


class Callback:
    def __init__(self):
        self.params = {}
        self.module = torch.nn.Module()

    def set_params(self, params: dict):
        self.params = params

    def set_module(self, module: torch.nn.Module):
        self.module = module

    def on_train_begin(self, image):
        pass

    def on_batch_end(self, log, batch):
        pass

    def on_epoch_end(self, log, image, epoch):
        pass


class CallbackList:
    def __init__(self, callbacks: [Callback] = None):
        self.callbacks = callbacks or []
        self.params = {}
        self.module = torch.nn.Module()

    def set_params(self, params: dict):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_module(self, module: torch.nn.Module):
        self.module = module
        for callback in self.callbacks:
            callback.set_module(module)

    def call_train_begin_hook(self, image):
        for callback in self.callbacks:
            callback.on_train_begin(image)

    def call_batch_end_hook(self, log, batch):
        for callback in self.callbacks:
            callback.on_batch_end(log, batch)

    def call_epoch_end_hook(self, log, image, epoch):
        for callback in self.callbacks:
            callback.on_epoch_end(log, image, epoch)


class Tensorboard(Callback):
    from torch.utils.tensorboard import SummaryWriter

    def __init__(self, file_path=None, per_batch=1):

        self.file_path = file_path
        self.per_batch = per_batch

        self.tb_writer = None

        if self.file_path is not None:
            check_and_mkdir(self.file_path)
            self.tb_writer = self.SummaryWriter(self.file_path)

        super().__init__()

    def on_train_begin(self, image):
        if self.tb_writer is not None:
            for i, k in enumerate(image, start=1):
                self.tb_writer.add_images(tag='init/' + k, img_tensor=normalize_(image[k]), global_step=0)

        if self.tb_writer is not None and 'config' in self.params:
            self.tb_writer.add_text(tag='config', text_string=(pprint.pformat(self.params['config']) + '\n').replace('\n', '\n\n'), global_step=0)

    def on_batch_end(self, log, batch):
        if self.tb_writer is not None:
            if batch % self.per_batch == 0:
                for i, k in enumerate(log, start=1):
                    self.tb_writer.add_scalar(tag='batch/' + k, scalar_value=log[k], global_step=batch)

    def on_epoch_end(self, log, image, epoch):
        if self.tb_writer is not None:
            for i, k in enumerate(log, start=1):
                self.tb_writer.add_scalar(tag='epoch/' + k, scalar_value=log[k], global_step=epoch)

            for i, k in enumerate(image, start=1):
                if image[k].shape.__len__() == 4 and (image[k].shape[1] == 1 or image[k].shape[1] == 3):
                    self.tb_writer.add_images(tag='epoch/' + k, img_tensor=normalize_(image[k]), global_step=epoch)


class BaseLogger(Callback):
    """Callback that accumulates epoch averages of metrics.
    This callback is automatically applied to every Keras model.
    # Arguments
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over an epoch.
            Metrics in this list will be logged as-is in `on_epoch_end`.
            All others will be averaged in `on_epoch_end`.
    """
    def __init__(self, file_path=None):
        super().__init__()

        self.__logger = logging.getLogger('main')
        formatter = logging.Formatter('%(asctime)s %(message)s')
        self.__logger.setLevel(logging.DEBUG)

        streamHandler = logging.StreamHandler()
        streamHandler.setFormatter(formatter)
        self.__logger.addHandler(streamHandler)

        if file_path is not None:
            check_and_mkdir(file_path)

            fileHandler = logging.FileHandler(file_path + 'logging.txt', mode='w')
            fileHandler.setFormatter(formatter)

            self.__logger.addHandler(fileHandler)

    def on_train_begin(self, image):
        structure, num_params = pformat_module(self.module)
        self.__logger.info("Module Structure \n\n" + structure + '\n')
        self.__logger.info('Module Params Amount: ' + num_params + '\n')

        try:
            self.__logger.info('Configuration \n\n' + pprint.pformat(self.params['config']) + '\n')
        except:
            self.__logger.info("CANNOT FIND CONFIG INFORMATION")

    def on_epoch_end(self, log, image, epoch):
        try:
            train_epoch = self.params['train_epoch']
            train_epoch = "%.3d" % train_epoch

        except:
            # Not Found in self.params[\'train_epoch\']
            train_epoch = "???"

        if log is not None:
            logger_pformat = '\n===============================\n' \
                             '  [BaseLogger] [%.3d/%s]\n' \
                             '===============================\n'  % (epoch, train_epoch)
            for k in log:
                logger_pformat += ('[%s]: [%.6f]\n' % (k, log[k]))

            self.__logger.critical(logger_pformat)


class CodeBackupER(Callback):
    def __init__(self, src_path=None, file_path=None):
        super().__init__()

        if (file_path is not None) and (src_path is not None):
            check_and_mkdir(file_path)

            MAX_CODE_SAVE = 100  # only 100 copies can be saved
            for i in range(MAX_CODE_SAVE):
                code_path = file_path + 'code%d/' % i
                if not os.path.exists(code_path):
                    shutil.copytree(src=src_path, dst=code_path)
                    break


class ModelCheckpoint(Callback):
    def __init__(self,
                 file_path = None,
                 monitors: [str] = None,
                 modes: [str] = None,
                 period: int = 10):

        super().__init__()

        self.file_path = file_path
        if self.file_path is not None:
            check_and_mkdir(self.file_path)

        self.monitors = monitors or []
        self.period = period

        self.monitor_ops = []
        self.best_epochs = []
        self.best_values = []

        modes = modes or []
        for mode in modes:
            if mode not in ['min', 'max']:
                warnings.warn('ModelCheckpoint mode %s is unknown' % mode, RuntimeWarning)

            if mode == 'min':
                self.monitor_ops.append(np.less)
                self.best_epochs.append(0)
                self.best_values.append(np.Inf)

            elif mode == 'max':
                self.monitor_ops.append(np.greater)
                self.best_epochs.append(0)
                self.best_values.append(-np.Inf)

        self.__logger = logging.getLogger('main')

    def on_epoch_end(self, log, image, epoch):
        try:
            state_dict = self.module.module.state_dict()
        except AttributeError:
            state_dict = self.module.state_dict()

        torch.save(state_dict, self.file_path + 'latest.pt')

        try:
            train_epoch = self.params['train_epoch']
            train_epoch = "%.3d" % train_epoch

        except:
            # Not Found in self.params[\'train_epoch\']
            train_epoch = "???"

        if (epoch % self.period == 0) and (self.file_path is not None):
            torch.save(state_dict, self.file_path + 'epoch%.3d.pt' % epoch)

        checkpoint_pformat = '\n===============================\n' \
                             '  [ModelCheckpoint] [%.3d/%s]\n' \
                             '===============================\n'  % (epoch, train_epoch)

        for i, monitor in enumerate(self.monitors):
            current = log.get(monitor)

            if self.monitor_ops[i](current, self.best_values[i]):
                checkpoint_pformat += '[%s] Improved: [%.5f] -> [%.5f] \n' % (
                    monitor, self.best_values[i], current)

                self.best_values[i] = current
                self.best_epochs[i] = epoch

                if self.file_path is not None:
                    torch.save(state_dict, self.file_path + 'best_%s.pt' % monitor)

            else:
                checkpoint_pformat += '[%s] Maintained: Current is [%.5f] Best is [%.5f] in Epoch [%.5d] \n' % (
                    monitor, current, self.best_values[i], self.best_epochs[i])

        self.__logger.critical(checkpoint_pformat)
