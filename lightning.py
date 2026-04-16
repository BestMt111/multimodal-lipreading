import torch
import torchaudio
from cosine import WarmupCosineScheduler
from datamodule.transforms import TextTransform

from pytorch_lightning import LightningModule
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus
from espnet.nets.scorers.ctc import CTCPrefixScorer
from scripts.change_state_dict import read_pretrained_model
from espnet.nets.lm_utils import get_model_conf, dynamic_import_lm, torch_load
from scripts.test_token_list import test_token_list
from scripts.process_str import process_str

def compute_WordorChar_level_distance(seq1, seq2, datatype):
    if datatype == 'en':
        return torchaudio.functional.edit_distance(seq1.lower().split(), seq2.lower().split())
    elif datatype == 'zh':
        seq1 = seq1.replace(" ", "")
        seq2 = seq2.replace(" ", "")
        return torchaudio.functional.edit_distance(list(seq1.replace('<unk>', '*')), list(seq2.replace('<unk>', '*')))


class ModelModule(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        if self.cfg.data.modality == "audio":
            self.backbone_args = self.cfg.model.audio_backbone
        elif self.cfg.data.modality == "video":
            self.backbone_args = self.cfg.model.visual_backbone

        self.text_transform = TextTransform(dict_path=self.cfg.data.dataset.dictionary_path)
        self.token_list = self.text_transform.token_list

        #传入了2个blank和2个EOS，
        # print(self.token_list[-1], self.token_list[-2])

        #测试token list有没有对上
        # test_token_list(token_list=self.token_list)

        self.model = E2E(len(self.token_list), self.backbone_args)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f'{total_params:,} total parameters.')
        self.language_type = cfg.data.dataset.language_type

        # -- initialise
        if self.cfg.pretrained_model_path:
            read_pretrained_model(cfg, self.model)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [{"name": "model", "params": self.model.parameters(), "lr": self.cfg.optimizer.lr}],
            weight_decay=self.cfg.optimizer.weight_decay,
            betas=(0.9, 0.98)
        )
        scheduler = WarmupCosineScheduler(
            optimizer,
            self.cfg.optimizer.warmup_epochs,
            self.cfg.trainer.max_epochs,
            len(self.trainer.datamodule.train_dataloader())
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def forward(self, sample):
        self.beam_search = get_beam_search_decoder(model=self.model, device=self.device, token_list=self.token_list)
        enc_feat, _ = self.model.encoder(sample.unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")
        return predicted

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, step_type="val")

    def test_step(self, sample, sample_idx):
        #sample为dict，包含两个key，一个是input，为真实的视频数据，放入模型中处理，另一个是target，代表label的tokenid
        enc_feat, _ = self.model.encoder(sample["input"].unsqueeze(0).to(self.device), None)
        enc_feat = enc_feat.squeeze(0)

        nbest_hyps = self.beam_search(enc_feat)
        nbest_hyps = [h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), 1)]]
        predicted_token_id = torch.tensor(list(map(int, nbest_hyps[0]["yseq"][1:])))
        predicted = self.text_transform.post_process(predicted_token_id).replace("<eos>", "")

        token_id = sample["target"]
        actual = self.text_transform.post_process(token_id)

        #处理字符串中的<unk>及空格
        actual, predicted = process_str(seq=actual), process_str(seq=predicted)

        #判断中文还是英文，edit_distance和length的计算逻辑不同
        if self.language_type == 'zh':
            # self.total_length += len(actual)
            self.total_length += len(list(actual.replace('\n', '')))
            self.total_edit_distance += compute_WordorChar_level_distance(actual, predicted, 'zh')
        elif self.language_type == 'en':
            self.total_length += len(actual.split())
            self.total_edit_distance += compute_WordorChar_level_distance(actual, predicted, 'en')
        else:
            raise RuntimeError('invalid language type')


        print('\nactual: ', actual)
        print('predicted: ', predicted)
        # print('total_edit_distance: ', self.total_edit_distance, '    total_length: ', self.total_length)
        print('cer: ', self.total_edit_distance / self.total_length)

        return

    def _step(self, batch, batch_idx, step_type):
        # input_lengths是一维的，代表视频的实际长度，padding_mask就是根据实际长度来进行生成的。
        loss, loss_ctc, loss_att, acc = self.model(batch["inputs"], batch["input_lengths"], batch["targets"])

        # 添加验证时进行打印的功能
        # loss, loss_ctc, loss_att, acc, pred_seq, truth_seq = self.model(batch["inputs"], batch["input_lengths"], batch["targets"])
        # self.predicted = self.text_transform.post_process(pred_seq).replace("<blank>", "").replace("<eos>", "")
        # self.actual = self.text_transform.post_process(truth_seq).replace("<blank>", "").replace("<eos>", "")
        # self.total_wer_editdistance += compute_WordorChar_level_distance(self.actual, self.predicted)
        # self.total_tokens += len(self.actual.split())
        # self.total_chars += len(list(self.actual))

        batch_size = len(batch["inputs"])

        if step_type == "train":
            self.log("loss", loss, on_step=True, on_epoch=True, batch_size=batch_size)
            self.log("loss_ctc", loss_ctc, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("loss_att", loss_att, on_step=False, on_epoch=True, batch_size=batch_size)
            self.log("decoder_acc", acc, on_step=True, on_epoch=True, batch_size=batch_size)
        else:
            self.log("loss_val", loss, batch_size=batch_size)
            self.log("loss_ctc_val", loss_ctc, batch_size=batch_size)
            self.log("loss_att_val", loss_att, batch_size=batch_size)
            self.log("decoder_acc_val", acc, batch_size=batch_size)

        if step_type == "train":
            self.log("monitoring_step", torch.tensor(self.global_step, dtype=torch.float32))

        return loss

    def on_train_epoch_start(self):
        sampler = self.trainer.train_dataloader.loaders.batch_sampler
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(self.current_epoch)
        return super().on_train_epoch_start()

    def on_test_epoch_start(self):
        print("\nTested on ", self.cfg.pretrained_model_path)
        self.total_length = 0
        self.total_edit_distance = 0
        if self.cfg.lm_model.enabled:
            self.beam_search = get_beam_search_decoder(model=self.model, device=self.device, token_list=self.token_list,
                                                       rnnlm=self.cfg.lm_model.rnnlm,
                                                       rnnlm_conf=self.cfg.lm_model.rnnlm_conf,
                                                       penalty=self.cfg.lm_model.penalty,
                                                       lm_weight=self.cfg.lm_model.lm_weight,
                                                       beam_size=self.cfg.lm_model.beam_size)
        else:
            self.beam_search = get_beam_search_decoder(model=self.model, device=self.device, token_list=self.token_list)

    def on_test_epoch_end(self):
        self.log("cer", self.total_edit_distance / self.total_length)


def get_beam_search_decoder(model, device='', token_list='', rnnlm=None, rnnlm_conf=None, penalty=0, ctc_weight=0.1, lm_weight=0., beam_size=40):
    if not rnnlm:
        lm = None
    else:
        lm_args = get_model_conf(rnnlm, rnnlm_conf)
        lm_model_module = getattr(lm_args, "model_module", "default")
        lm_class = dynamic_import_lm(lm_model_module, lm_args.backend)
        lm = lm_class(len(token_list), lm_args).to(device)
        torch_load(rnnlm, lm)
        lm.eval()

    scorers = {
        "decoder": model.decoder,
        "ctc": CTCPrefixScorer(model.ctc, model.eos),
        "length_bonus": LengthBonus(len(token_list)),
        "lm": lm
    }

    weights = {
        "decoder": 1.0 - ctc_weight,
        "ctc": ctc_weight,
        "lm": lm_weight,
        "length_bonus": penalty,
    }

    # scorers = {
    #     "decoder": model.decoder,
    #     "ctc": CTCPrefixScorer(model.ctc, model.eos),
    #     "length_bonus": LengthBonus(len(token_list)),
    #     "lm": None
    # }
    #
    # weights = {
    #     "decoder": 1.0 - ctc_weight,
    #     "ctc": ctc_weight,
    #     "lm": 0.0,
    #     "length_bonus": 0.0,
    # }

    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=model.sos,
        eos=model.eos,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )