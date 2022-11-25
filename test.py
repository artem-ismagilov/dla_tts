import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import hw_tts.model as module_model
from hw_tts.trainer import Trainer
from hw_tts.utils import ROOT_PATH
from hw_tts.utils.object_loading import get_dataloaders
from hw_tts.utils.parse_config import ConfigParser
from hw_tts.metric.utils import calc_wer, calc_cer

DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_file, args):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # text_encoder
    text_encoder = config.get_text_encoder()

    # setup data_loader instances
    dataloaders = get_dataloaders(config, text_encoder)

    # build model architecture
    model = config.init_obj(config["arch"], module_model, n_class=len(text_encoder))
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    results = []

    wers, cers = [], []
    wers_argmax, cers_argmax = [], []
    with torch.no_grad():
        for batch_num, batch in enumerate(tqdm(dataloaders["test"])):
            batch = Trainer.move_batch_to_device(batch, device)
            output = model(**batch)
            if type(output) is dict:
                batch.update(output)
            else:
                batch["logits"] = output
            batch["log_probs"] = torch.log_softmax(batch["logits"], dim=-1)
            batch["log_probs_length"] = model.transform_input_lengths(
                batch["spectrogram_length"]
            )
            batch["argmax"] = batch["log_probs"].argmax(-1)
            for i in range(len(batch["text"])):
                argmax = batch["argmax"][i]
                argmax = argmax[: int(batch["log_probs_length"][i])]
                text_argmax = text_encoder.ctc_decode(argmax.cpu().numpy())

                beam_search_result = text_encoder.ctc_beam_search(
                    batch["log_probs"][i],
                    batch["log_probs_length"][i],
                    beam_size=100,
                    result_size=args.hyps_size,
                    impl_type=args.bs_impl_type)

                wers.append(calc_wer(batch["text"][i].strip().lower(), beam_search_result[0].text))
                cers.append(calc_cer(batch["text"][i].strip().lower(), beam_search_result[0].text))

                wers_argmax.append(calc_wer(batch["text"][i].strip().lower(), text_argmax))
                cers_argmax.append(calc_cer(batch["text"][i].strip().lower(), text_argmax))

                results.append(
                    {
                        "ground_trurh": batch["text"][i],
                        "pred_text_argmax": text_argmax,
                        "pred_text_beam_search": beam_search_result
                    }
                )

    print()
    print('Mean WER (argmax): {:.3f}, Mean CER (argmax): {:.3f}'.format(sum(wers_argmax) / len(wers_argmax), sum(cers_argmax) / len(cers_argmax)))
    print('Mean WER (beam search): {:.3f}, Mean CER (beam search): {:.3f}'.format(sum(wers) / len(wers), sum(cers) / len(cers)))

    with Path(out_file).open("w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-o",
        "--output",
        default="output.json",
        type=str,
        help="File to write results (.json)",
    )
    args.add_argument(
        "-t",
        "--test-data-folder",
        default=None,
        type=str,
        help="Path to dataset",
    )
    args.add_argument(
        "-b",
        "--batch-size",
        default=20,
        type=int,
        help="Test dataset batch size",
    )
    args.add_argument(
        "-j",
        "--jobs",
        default=1,
        type=int,
        help="Number of workers for test dataloader",
    )
    args.add_argument(
        "-s",
        "--hyps-size",
        default=1,
        type=int,
        help="Returned hypothesis count",
    )
    args.add_argument(
        "-i",
        "--bs-impl-type",
        default='library',
        type=str,
        help="Beam search implementation",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    # if `--test-data-folder` was provided, set it as a default test set
    if args.test_data_folder is not None:
        test_data_folder = Path(args.test_data_folder).absolute().resolve()
        assert test_data_folder.exists()
        config.config["data"] = {
            "test": {
                "batch_size": args.batch_size,
                "num_workers": args.jobs,
                "datasets": [
                    {
                        "type": "CustomDirAudioDataset",
                        "args": {
                            "audio_dir": str(test_data_folder / "audio"),
                            "transcription_dir": str(
                                test_data_folder / "transcriptions"
                            ),
                        },
                    }
                ],
            }
        }

    assert config.config.get("data", {}).get("test", None) is not None
    config["data"]["test"]["batch_size"] = args.batch_size
    config["data"]["test"]["n_jobs"] = args.jobs

    main(config, args.output, args)