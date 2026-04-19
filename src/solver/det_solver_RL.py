"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time
import json
import datetime

import torch

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine_TTA_selfT_RL import train_one_epoch_eval, evaluate
from ..token.freeze_bak import _ycfg, freeze_backbone_light_train, unfreeze_all

from ..RL.rl_adaptation_controller import RLAdaptationController, RewardEMA


def build_rl_controller(args, device):
    rl_state_dim = int(_ycfg(args, ["rl_state_dim"], 16))
    rl_hidden_dim = int(_ycfg(args, ["rl_hidden_dim"], 64))
    rl_lr = float(_ycfg(args, ["rl_lr"], 1e-4))
    rl_weight_decay = float(_ycfg(args, ["rl_weight_decay"], 1e-4))
    rl_reward_momentum = float(_ycfg(args, ["rl_reward_momentum"], 0.9))

    rl_controller = RLAdaptationController(
        state_dim=rl_state_dim,
        hidden_dim=rl_hidden_dim,
    ).to(device)

    rl_optimizer = torch.optim.Adam(
        rl_controller.parameters(),
        lr=rl_lr,
        weight_decay=rl_weight_decay,
    )

    reward_ema = RewardEMA(momentum=rl_reward_momentum)
    return rl_controller, rl_optimizer, reward_ema


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")
    return total_params, trainable_params


class DetSolver(BaseSolver):

    def fit(self, ):

        print("Start training")
        self.train()
        args = self.cfg

        self.rl_enable = bool(_ycfg(args, ["rl_enable"], False))

        if self.rl_enable:
            self.rl_controller, self.rl_optimizer, self.reward_ema = build_rl_controller(args, self.device)
        else:
            self.rl_controller, self.rl_optimizer, self.reward_ema = None, None, None

        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        best_stat = {'epoch': -1, }

        start_time = time.time()
        start_epcoch = self.last_epoch + 1

        total_epochs = self.cfg.epoches

        count_parameters(self.model)

        self.freeze_epochs = int(_ycfg(self.cfg, ["token.freeze_epochs"], 2))

        print('[freeze epoch]:', self.freeze_epochs)
        freeze_backbone_light_train(
            self.model,
            keep_modules=("encoder.ra_modules", "decoder", "head"),
            verbose=True
        )

        for epoch in range(start_epcoch, args.epoches):

            self.train_dataloader.set_epoch(epoch)
            # self.train_dataloader.dataset.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)


            # —— 到达解冻点：放开全部参数，并重建优化器（可保留动量/EMA由你原逻辑处理）——
            if epoch == self.freeze_epochs:
                unfreeze_all(self.model, verbose=True)

            train_stats = train_one_epoch_eval(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.val_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,
                print_freq=args.print_freq,
                ema=self.ema,
                scaler=self.scaler,
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer,
                postprocessor=self.postprocessor,
                coco_evaluator=self.evaluator,
                total_epochs=total_epochs,

                rl_enable=self.rl_enable,
                rl_controller=self.rl_controller,
                rl_optimizer=self.rl_optimizer,
                reward_ema=self.reward_ema,
                rl_entropy_coef=float(_ycfg(args, ["rl_entropy_coef"], 0.01)),
                rl_state_dim=int(_ycfg(args, ["rl_state_dim"], 16)),
                rl_greedy_eval=bool(_ycfg(args, ["rl_greedy_eval"], False)),
            )

            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            self.last_epoch += 1

            if self.output_dir:
                checkpoint_paths = [self.output_dir / 'last.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            # module = self.ema.module if self.ema else self.model
            module = self.model

            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            module1 = self.ema.module if self.ema else self.model

            test_stats, coco_evaluator = evaluate(
                module1,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            # TODO
            for k in test_stats:
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}'.format(k), v, epoch)

                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # for evaluation logs
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                       self.output_dir / "eval" / name)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        self.eval()

        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                                              self.val_dataloader, self.evaluator, self.device)

        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return
