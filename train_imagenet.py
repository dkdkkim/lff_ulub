import os
import pickle
from tqdm import tqdm
from datetime import datetime

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from config import ex
from data.util import get_dataset, IdxDataset, ZippedDataset
from module.loss import GeneralizedCELoss
from module.util import get_model
from util import MultiDimAverageMeter, EMA

from loader_ulub.imagenet_dataset import get_imagenet_dataloader
from option import get_option

def _validate_imagenet(data_loader, step=0, valid_type='',
                        num_clusters=9,
                        num_cluster_repeat=3,
                        key=None):
        self._mode_setting(is_train=False)

        if not self.option.is_train:
            print("not in training process")
            self._initialization()
            if self.option.checkpoint_orth is not None:
                self._load_model()
            else:
                print("No trained model")
                sys.exit()

        total_num_correct_orth = 0.
        total_num_test = 0.
        total_loss_orth = 0.
        total_loss_conv = 0.
        total_loss_trans = 0.

        total = 0
        f_correct = 0
        num_correct = [np.zeros([self.option.n_class, num_clusters]) for _ in range(num_cluster_repeat)]
        num_instance = [np.zeros([self.option.n_class, num_clusters]) for _ in range(num_cluster_repeat)]

        for i, (images, labels, bias_labels) in enumerate(tqdm(data_loader)):

            images = self._get_variable(images)
            labels = self._get_variable(labels)

            batch_size = images.shape[0]
            total_num_test += batch_size
            total += batch_size
            if self.option.ubnet:
                self.optim_orth.zero_grad()
                out = self.extract_features(images)

                pred_label_orth, loss_conv, loss_trans = self.ubnet(out)

                if key == 'unbiased':
                    num_correct, num_instance = self.imagenet_unbiased_accuracy(pred_label_orth.data, labels,
                                                                                bias_labels,
                                                                                num_correct, num_instance,
                                                                                num_cluster_repeat)
                else:
                    f_correct += self.n_correct(pred_label_orth, labels)

                loss_orth = self.loss_ubnet(pred_label_orth, torch.squeeze(labels))
                total_num_correct_orth += self._num_correct(pred_label_orth, labels, topk=1).data
                total_loss_orth += loss_orth.data * batch_size
                total_loss_conv += loss_conv
                total_loss_trans += loss_trans

        if self.option.ubnet:
            if key == 'unbiased':
                for k in range(num_cluster_repeat):
                    x, y = [], []
                    _num_correct, _num_instance = num_correct[k].flatten(), num_instance[k].flatten()
                    for i in range(_num_correct.shape[0]):
                        __num_correct, __num_instance = _num_correct[i], _num_instance[i]
                        if __num_instance >= 10:
                            x.append(__num_instance)
                            y.append(__num_correct / __num_instance)
                    f_correct += sum(y) / len(x)

                avg_acc_orth = f_correct / num_cluster_repeat
            else:
                avg_acc_orth = f_correct / total

            avg_loss_orth = total_loss_orth / total_num_test
            msg = f"[EVALUATION] {key} step{step} LOSS : {avg_loss_orth}, ACCURACY : {avg_acc_orth} LOSS_CONV : {total_loss_conv / total_num_test} LOSS_TRANS : {total_loss_trans / total_num_test}"
            self.writer.add_scalars('Loss/epoch', {f'valid_{valid_type}': avg_loss_orth}, step)
            self.writer.add_scalars('Accuracy/epoch', {f'valid_{valid_type}': avg_acc_orth}, step)
            self.cur_acc_orth = avg_acc_orth
        self.logger.info(msg)
# @ex.automain
def train(
    main_tag,
    dataset_tag,
    model_tag,
    data_dir,
    log_dir,
    device,
    target_attr_idx,
    bias_attr_idx,
    main_num_steps,
    main_valid_freq,
    main_batch_size,
    main_optimizer_tag,
    main_learning_rate,
    main_weight_decay,
):
    print(main_tag)
    print(dataset_tag)
    
    device = torch.device(device)
    start_time = datetime.now()
    writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag))
    train_dataset = get_dataset(
        dataset_tag,
        data_dir=data_dir,
        dataset_split="train",
        transform_split="train",
    )

    train_target_attr = train_dataset.attr[:, target_attr_idx]
    train_bias_attr = train_dataset.attr[:, bias_attr_idx]
    attr_dims = []
    attr_dims.append(torch.max(train_target_attr).item() + 1)
    attr_dims.append(torch.max(train_bias_attr).item() + 1)
    num_classes = attr_dims[0]
        
    train_dataset = IdxDataset(train_dataset)    

    # make loader    
    train_loader = DataLoader(
        train_dataset,
        batch_size=main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    valid_loader= get_imagenet_dataloader(root='/data/dk/ulub/dataset/imagenet/val',
                                            batch_size=main_batch_size,
                                            train=False,
                                            val_data='ImageNet')

    # define model and optimizer
    model_b = get_model(model_tag, attr_dims[0]).to(device)
    model_d = get_model(model_tag, attr_dims[0]).to(device)
    
    if main_optimizer_tag == "SGD":
        optimizer_b = torch.optim.SGD(
            model_b.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
            momentum=0.9,
        )
        optimizer_d = torch.optim.SGD(
            model_d.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
            momentum=0.9,
        )
    elif main_optimizer_tag == "Adam":
        optimizer_b = torch.optim.Adam(
            model_b.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
        optimizer_d = torch.optim.Adam(
            model_d.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
    elif main_optimizer_tag == "AdamW":
        optimizer_b = torch.optim.AdamW(
            model_b.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
        optimizer_d = torch.optim.AdamW(
            model_d.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
    else:
        raise NotImplementedError
    
    # define loss
    criterion = nn.CrossEntropyLoss(reduction='none')
    bias_criterion = GeneralizedCELoss()
    
    sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), alpha=0.7)
    sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), alpha=0.7)

    def imagenet_unbiased_accuracy(outputs, labels, cluster_labels,
                                   num_correct, num_instance,
                                   num_cluster_repeat=3):
        for j in range(num_cluster_repeat):
            for i in range(outputs.size(0)):
                output = outputs[i]
                label = labels[i]
                cluster_label = cluster_labels[j][i]

                _, pred = output.topk(1, 0, largest=True, sorted=True)
                correct = pred.eq(label).view(-1).float()

                num_correct[j][label][cluster_label] += correct.item()
                num_instance[j][label][cluster_label] += 1

        return num_correct, num_instance
    # define evaluation function
    def evaluate(model, data_loader, key=None, n_class=9, num_clusters=9, num_cluster_repeat=3):
        model.eval()

        total_num_test = 0.

        total = 0
        f_correct = 0
        num_correct = [np.zeros([n_class, num_clusters]) for _ in range(num_cluster_repeat)]
        num_instance = [np.zeros([n_class, num_clusters]) for _ in range(num_cluster_repeat)]

        for i, (images, labels, bias_labels) in enumerate(tqdm(data_loader)):

            images = images.to(device)
            labels = labels.to(device)

            batch_size = images.shape[0]
            total_num_test += batch_size
            total += batch_size

            with torch.no_grad():
                logit = model(images)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)

            if key == 'unbiased':
                num_correct, num_instance = imagenet_unbiased_accuracy(logit, labels,
                                                                            bias_labels,
                                                                            num_correct, num_instance,
                                                                            num_cluster_repeat)
            else:
                f_correct += (pred == labels).sum().item()

        if key == 'unbiased':
            for k in range(num_cluster_repeat):
                x, y = [], []
                _num_correct, _num_instance = num_correct[k].flatten(), num_instance[k].flatten()
                for i in range(_num_correct.shape[0]):
                    __num_correct, __num_instance = _num_correct[i], _num_instance[i]
                    if __num_instance >= 10:
                        x.append(__num_instance)
                        y.append(__num_correct / __num_instance)
                f_correct += sum(y) / len(x)

            avg_acc_orth = f_correct / num_cluster_repeat
        else:
            avg_acc_orth = f_correct / total
        model.train()
        return avg_acc_orth

    # jointly training biased/de-biased model
    valid_accs_list_biased, valid_accs_list_unbiased = [], []
    num_updated = 0
    
    for step in tqdm(range(main_num_steps)):
        
        # train main model
        try:
            index, data, attr = next(train_iter)
        except:
            train_iter = iter(train_loader)
            index, data, attr = next(train_iter)

        data = data.to(device)
        attr = attr.to(device)
        label = attr[:, target_attr_idx]
        bias_label = attr[:, bias_attr_idx]
        
        logit_b = model_b(data)
        if np.isnan(logit_b.mean().item()):
            print(logit_b)
            raise NameError('logit_b')
        logit_d = model_d(data)
        
        loss_b = criterion(logit_b, label).cpu().detach()
        loss_d = criterion(logit_d, label).cpu().detach()
                
        if np.isnan(loss_b.mean().item()):
            raise NameError('loss_b')
        if np.isnan(loss_d.mean().item()):
            raise NameError('loss_d')
        
        loss_per_sample_b = loss_b
        loss_per_sample_d = loss_d
        
        # EMA sample loss
        sample_loss_ema_b.update(loss_b, index)
        sample_loss_ema_d.update(loss_d, index)
        
        # class-wise normalize
        loss_b = sample_loss_ema_b.parameter[index].clone().detach()
        loss_d = sample_loss_ema_d.parameter[index].clone().detach()
        
        if np.isnan(loss_b.mean().item()):
            raise NameError('loss_b_ema')
        if np.isnan(loss_d.mean().item()):
            raise NameError('loss_d_ema')
        
        label_cpu = label.cpu()
        
        for c in range(num_classes):
            class_index = np.where(label_cpu == c)[0]
            max_loss_b = sample_loss_ema_b.max_loss(c)
            max_loss_d = sample_loss_ema_d.max_loss(c)
            loss_b[class_index] /= max_loss_b
            loss_d[class_index] /= max_loss_d
            
        # re-weighting based on loss value / generalized CE for biased model
        loss_weight = loss_b / (loss_b + loss_d + 1e-8)
        if np.isnan(loss_weight.mean().item()):
            raise NameError('loss_weight')
            
        loss_b_update = bias_criterion(logit_b, label)

        if np.isnan(loss_b_update.mean().item()):
            raise NameError('loss_b_update')
        loss_d_update = criterion(logit_d, label) * loss_weight.to(device)
        if np.isnan(loss_d_update.mean().item()):
            raise NameError('loss_d_update')
        loss = loss_b_update.mean() + loss_d_update.mean()
        
        num_updated += loss_weight.mean().item() * data.size(0)

        optimizer_b.zero_grad()
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_b.step()
        optimizer_d.step()
        
        main_log_freq = 10
        if step % main_log_freq == 0:
        
            writer.add_scalar("loss/b_train", loss_per_sample_b.mean(), step)
            writer.add_scalar("loss/d_train", loss_per_sample_d.mean(), step)

            bias_attr = attr[:, bias_attr_idx]

            aligned_mask = (label == bias_attr)
            skewed_mask = (label != bias_attr)
            
            writer.add_scalar('loss_variance/b_ema', sample_loss_ema_b.parameter.var(), step)
            writer.add_scalar('loss_std/b_ema', sample_loss_ema_b.parameter.std(), step)
            writer.add_scalar('loss_variance/d_ema', sample_loss_ema_d.parameter.var(), step)
            writer.add_scalar('loss_std/d_ema', sample_loss_ema_d.parameter.std(), step)

            if aligned_mask.any().item():
                writer.add_scalar("loss/b_train_aligned", loss_per_sample_b[aligned_mask].mean(), step)
                writer.add_scalar("loss/d_train_aligned", loss_per_sample_d[aligned_mask].mean(), step)
                writer.add_scalar('loss_weight/aligned', loss_weight[aligned_mask].mean(), step)

            if skewed_mask.any().item():
                writer.add_scalar("loss/b_train_skewed", loss_per_sample_b[skewed_mask].mean(), step)
                writer.add_scalar("loss/d_train_skewed", loss_per_sample_d[skewed_mask].mean(), step)
                writer.add_scalar('loss_weight/skewed', loss_weight[skewed_mask].mean(), step)

        if step % main_valid_freq == 0:
            valid_accs_b_unbiased = evaluate(model_b, valid_loader, 'unbiased')
            valid_accs_b_biased = evaluate(model_b, valid_loader, 'biased')
            valid_accs_d_unbiased = evaluate(model_d, valid_loader, 'unbiased')
            valid_accs_d_biased = evaluate(model_d, valid_loader, 'biased')
            valid_accs_list_unbiased.append(valid_accs_d_unbiased)
            valid_accs_list_biased.append(valid_accs_d_biased)
            writer.add_scalar("acc/b_valid_unbiased", valid_accs_b_unbiased, step)
            writer.add_scalar("acc/b_valid_biased", valid_accs_b_biased, step)
            writer.add_scalar("acc/d_valid_unbiased", valid_accs_d_unbiased, step)
            writer.add_scalar("acc/d_valid_biased", valid_accs_d_biased, step)         
                        
            num_updated_avg = num_updated / main_batch_size / main_valid_freq
            writer.add_scalar("num_updated/all", num_updated_avg, step)
            num_updated = 0

    os.makedirs(os.path.join(log_dir, "result", main_tag), exist_ok=True)
    result_path = os.path.join(log_dir, "result", main_tag, "result.th")
    model_path = os.path.join(log_dir, "result", main_tag, "model.th")
    valid_attrwise_accs_list_ub1 = torch.stack(valid_attrwise_accs_list_ub1)
    valid_attrwise_accs_list_ub2 = torch.stack(valid_attrwise_accs_list_ub2)
    with open(result_path, "wb") as f:
        torch.save({"valid/attrwise_accs_ub1": valid_attrwise_accs_list_ub1,
                    "valid/attrwize_accs_ub2": valid_attrwise_accs_list_ub2}, f)
    state_dict = {
        'steps': step, 
        'state_dict': model_d.state_dict(), 
        'optimizer': optimizer_d.state_dict(), 
    }
    with open(model_path, "wb") as f:
        torch.save(state_dict, f)
    

if __name__ == '__main__':
    option = get_option()
    for i in range(1,4):
        train(main_tag=f'imagenet_{option.exp_name}_0{i}',
                dataset_tag='imagenet',
                model_tag='ResNet18',
                data_dir=option.data_path,
                log_dir=option.log_path,
                device=option.gpu,
                target_attr_idx=0,
                bias_attr_idx=0,
                main_num_steps = 213 * option.epochs, # 88 * epoch
                main_valid_freq = 213,
                main_batch_size = 256,
                main_learning_rate = option.lr_init,
                main_weight_decay = 1e-4,
                main_optimizer_tag='Adam'
                )