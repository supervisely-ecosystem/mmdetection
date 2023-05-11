import datetime
from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.text import TextLoggerHook
import supervisely_lib as sly
from sly_train_progress import add_progress_to_request
import sly_globals as g
import classes as cls


@HOOKS.register_module()
class SuperviselyLoggerHook(TextLoggerHook):
    def __init__(
        self, by_epoch=True, interval=10, ignore_last=True, reset_flag=False, interval_exp_name=1000
    ):
        super(SuperviselyLoggerHook, self).__init__(
            by_epoch, interval, ignore_last, reset_flag, interval_exp_name
        )
        self.progress_epoch = None
        self.progress_iter = None
        self._lrs = []

    def _log_info(self, log_dict, runner):
        super(SuperviselyLoggerHook, self)._log_info(log_dict, runner)

        if log_dict["mode"] == "train" and "time" in log_dict.keys():
            self.time_sec_tot += log_dict["time"] * self.interval
            time_sec_avg = self.time_sec_tot / (runner.iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (len(runner.data_loader) * runner.max_epochs - runner.iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            log_dict["eta"] = eta_str

        if self.progress_epoch is None:
            self.progress_epoch = sly.Progress("Epochs", runner.max_epochs)
        if self.progress_iter is None:
            self.progress_iter = sly.Progress("Iterations", len(runner.data_loader))

        fields = []
        if log_dict["mode"] == "train":
            self.progress_epoch.set_current_value(log_dict["epoch"])
            if log_dict["iter"] % len(runner.data_loader) == 0:
                progress_iter_value = float(self.progress_iter.total)
            else:
                progress_iter_value = log_dict["iter"] % len(runner.data_loader)

            self.progress_iter.set(progress_iter_value, len(runner.data_loader))
            fields.append({"field": "data.eta", "payload": log_dict["eta"]})

        fields.append({"field": "state.isValidation", "payload": log_dict["mode"] == "val"})

        add_progress_to_request(fields, "Epoch", self.progress_epoch)
        add_progress_to_request(fields, "Iter", self.progress_iter)

        if log_dict["mode"] == "train":
            epoch_float = float(self.progress_epoch.current) + float(
                self.progress_iter.current
            ) / float(self.progress_iter.total)
            loss_names = [
                "loss_bbox",
                "loss_cls",
                "loss_mask",
                "loss_iou",
                "loss_rpn_cls",
                "loss_rpn_bbox",
                "loss_semantic_seg",
                "loss_other",
            ]
            basic_loss_names = loss_names[:4]
            other_loss_names = loss_names[4:]

            # aggregate losses
            for loss_name in loss_names[:-1]:
                if loss_name in log_dict.keys():
                    continue
                losses = []
                for key, val in log_dict.items():
                    if key.endswith(loss_name):
                        losses.append(val)
                if losses:
                    log_dict[loss_name] = sum(losses)

            # aggregate other losses
            other_losses = []
            for key, val in log_dict.items():
                losses = []
                for loss_name in loss_names[:-1]:
                    if key.endswith(loss_name):
                        losses.append(key)
                if not losses and "loss" in key and key != "loss":
                    other_losses.append(val)
            if other_losses:
                log_dict["loss_other"] = sum(other_losses)

            fields.extend(
                [
                    {
                        "field": "state.chartLR.series[0].data",
                        "payload": [[epoch_float, round(log_dict["lr"], 6)]],
                        "append": True,
                    },
                    {
                        "field": "state.chartLossBasic.series[0].data",
                        "payload": [[epoch_float, round(log_dict["loss"], 6)]],
                        "append": True,
                    },
                ]
            )
            for idx, loss_name in enumerate(basic_loss_names):
                if loss_name in log_dict.keys():
                    fields.append(
                        {
                            "field": f"state.chartLossBasic.series[{idx + 1}].data",
                            "payload": [[epoch_float, round(log_dict[loss_name], 6)]],
                            "append": True,
                        }
                    )

            for idx, loss_name in enumerate(other_loss_names):
                if loss_name in log_dict.keys():
                    fields.append(
                        {
                            "field": f"state.chartLossOther.series[{idx}].data",
                            "payload": [[epoch_float, round(log_dict[loss_name], 6)]],
                            "append": True,
                        }
                    )

            if "time" in log_dict.keys():
                fields.extend(
                    [
                        {
                            "field": "state.chartTime.series[0].data",
                            "payload": [[epoch_float, log_dict["time"]]],
                            "append": True,
                        },
                        {
                            "field": "state.chartDataTime.series[0].data",
                            "payload": [[epoch_float, log_dict["data_time"]]],
                            "append": True,
                        },
                        {
                            "field": "state.chartMemory.series[0].data",
                            "payload": [[epoch_float, log_dict["memory"]]],
                            "append": True,
                        },
                    ]
                )
        if log_dict["mode"] == "val":
            for class_ind, class_name in enumerate(cls.selected_classes):
                if f"bbox_AP_{class_name}" in log_dict.keys():
                    fields.append(
                        {
                            "field": f"state.chartBoxClassAP.series[{class_ind}].data",
                            "payload": [[log_dict["epoch"], log_dict[f"bbox_AP_{class_name}"]]],
                            "append": True,
                        }
                    )
                else:
                    sly.logger.warn(f"bbox_AP_{class_name} not found in log dictionary")
                if f"segm_AP_{class_name}" in log_dict.keys():
                    fields.append(
                        {
                            "field": f"state.chartMaskClassAP.series[{class_ind}].data",
                            "payload": [[log_dict["epoch"], log_dict[f"segm_AP_{class_name}"]]],
                            "append": True,
                        }
                    )
            if "bbox_mAP" in log_dict.keys():
                fields.append(
                    {
                        "field": f"state.chartMAP.series[0].data",
                        "payload": [[log_dict["epoch"], log_dict[f"bbox_mAP"]]],
                        "append": True,
                    }
                )
            if f"segm_mAP" in log_dict.keys():
                fields.append(
                    {
                        "field": f"state.chartMAP.series[1].data",
                        "payload": [[log_dict["epoch"], log_dict[f"segm_mAP"]]],
                        "append": True,
                    }
                )
        try:
            g.api.app.set_fields(g.task_id, fields)
        except Exception as e:
            print("Unabled to write metrics to chart!")
            print(e)
