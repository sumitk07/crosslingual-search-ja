"""Machine translation of questions with NLLB-200, with a cache on disk.

The cache means a second run does not translate again (translation is the slowest step).
"""
import json
import os

NLLB = {"ja": "jpn_Jpan", "en": "eng_Latn", "hi": "hin_Deva"}


class Translator:
    def __init__(self, model_name, cache_path, device="cpu", beams=4, batch_size=16):
        self.model_name, self.device, self.beams, self.bs = model_name, device, beams, batch_size
        self.cache_path = cache_path
        self.cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
        self._tok = self._model = None

    def _load(self):
        if self._model is None:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            print(f"Loading translation model {self.model_name} (first time downloads about 2.5 GB)...")
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device).eval()

    def translate(self, texts, src, tgt):
        key = lambda t: f"{src}>{tgt}|{t}"
        todo = [t for t in dict.fromkeys(texts) if key(t) not in self.cache]
        if todo:
            import torch
            self._load()
            self._tok.src_lang = NLLB[src]
            tgt_id = self._tok.convert_tokens_to_ids(NLLB[tgt])
            for i in range(0, len(todo), self.bs):
                batch = todo[i:i + self.bs]
                enc = self._tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
                with torch.no_grad():
                    out = self._model.generate(**enc, forced_bos_token_id=tgt_id, num_beams=self.beams,
                                               max_new_tokens=128)
                for t, o in zip(batch, self._tok.batch_decode(out, skip_special_tokens=True)):
                    self.cache[key(t)] = o
                print(f"  translated {min(i + self.bs, len(todo))}/{len(todo)} ({src}->{tgt})")
                self.save()
        return [self.cache[key(t)] for t in texts]

    def save(self):
        json.dump(self.cache, open(self.cache_path, "w", encoding="utf-8"), ensure_ascii=False, indent=0)


class FakeTranslator:
    """For --synthetic tests only: marks the text instead of translating it."""

    def translate(self, texts, src, tgt):
        return [f"[{src}>{tgt}] {t}" if tgt != "ja" else t.split("] ", 1)[-1] for t in texts]
