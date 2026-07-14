"""Offline OpenAI CLIP BPE tokenizer used by the EM-CLIP text branch.

The vocabulary asset is vendored from the MIT-licensed OpenAI CLIP repository.
Keeping tokenization local prevents training from silently substituting arbitrary
token ids for the pretrained CLIP embedding table.
"""

import gzip
import html
import os
import unicodedata
from functools import lru_cache


@lru_cache()
def bytes_to_unicode():
    """Return the reversible byte-to-unicode table used by OpenAI CLIP."""
    byte_values = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    unicode_values = byte_values[:]
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            unicode_values.append(256 + extra)
            extra += 1
    return dict(zip(byte_values, (chr(value) for value in unicode_values)))


def get_pairs(word):
    """Return adjacent symbol pairs for one BPE word tuple."""
    pairs = set()
    previous = word[0]
    for character in word[1:]:
        pairs.add((previous, character))
        previous = character
    return pairs


def _clean_text(text):
    try:
        import ftfy
    except ImportError:
        # Dataset class labels are expected to be valid Unicode. NFKC preserves
        # exact behavior for normal ASCII labels while still normalizing common
        # compatibility characters when ftfy is not installed.
        text = unicodedata.normalize("NFKC", text)
    else:
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return " ".join(text.strip().split())


class OpenAIClipBPETokenizer:
    """Exact CLIP BPE ids plus label-token spans for prompt construction."""

    pad_id = 0

    def __init__(self, bpe_path=None, context_length=77):
        try:
            import regex
        except ImportError as exc:
            raise ImportError(
                "OpenAI CLIP tokenization requires the 'regex' package. "
                "Install the dependency in the existing environment without changing PyTorch."
            ) from exc

        if bpe_path is None:
            bpe_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")
        bpe_path = os.path.abspath(os.path.expanduser(bpe_path))
        if not os.path.isfile(bpe_path):
            raise FileNotFoundError("CLIP BPE vocabulary does not exist: %s" % bpe_path)

        self.context_length = int(context_length)
        self.byte_encoder = bytes_to_unicode()
        with gzip.open(bpe_path, "rt", encoding="utf-8") as handle:
            merges = handle.read().split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        merges = [tuple(merge.split()) for merge in merges if merge]

        vocab = list(self.byte_encoder.values())
        vocab.extend(value + "</w>" for value in self.byte_encoder.values())
        vocab.extend("".join(merge) for merge in merges)
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }
        self.pattern = regex.compile(
            r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )
        self.sot_id = self.encoder["<|startoftext|>"]
        self.eot_id = self.encoder["<|endoftext|>"]
        self.vocab_size = len(self.encoder)
        if (self.vocab_size, self.sot_id, self.eot_id) != (49408, 49406, 49407):
            raise RuntimeError(
                "unexpected CLIP BPE vocabulary layout: vocab=%d SOT=%d EOT=%d"
                % (self.vocab_size, self.sot_id, self.eot_id)
            )

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)
        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            merged = []
            index = 0
            while index < len(word):
                try:
                    next_index = word.index(first, index)
                except ValueError:
                    merged.extend(word[index:])
                    break
                merged.extend(word[index:next_index])
                index = next_index
                if index < len(word) - 1 and word[index] == first and word[index + 1] == second:
                    merged.append(first + second)
                    index += 2
                else:
                    merged.append(word[index])
                    index += 1
            word = tuple(merged)
            if len(word) == 1:
                break
            pairs = get_pairs(word)
        result = " ".join(word)
        self.cache[token] = result
        return result

    def encode(self, text):
        bpe_tokens = []
        text = _clean_text(text).lower()
        for token in self.pattern.findall(text):
            encoded = "".join(self.byte_encoder[value] for value in token.encode("utf-8"))
            bpe_tokens.extend(self.encoder[piece] for piece in self.bpe(encoded).split(" "))
        return bpe_tokens

    def encode_prompt(self, template, label):
        if template.count("{}") != 1:
            raise ValueError("prompt template must contain exactly one '{}': %s" % template)
        prefix, suffix = template.split("{}", 1)
        prefix_ids = self.encode(prefix)
        label_ids = self.encode(label)
        suffix_ids = self.encode(suffix)
        if not label_ids:
            raise ValueError("class label must contain at least one CLIP BPE token: %r" % label)

        prompt_ids = self.encode(template.format(label))
        combined_ids = prefix_ids + label_ids + suffix_ids
        if prompt_ids != combined_ids:
            raise ValueError(
                "cannot derive an exact label-token span for template=%r label=%r; "
                "put whitespace around the '{}' placeholder" % (template, label)
            )

        token_ids = [self.sot_id] + combined_ids + [self.eot_id]
        if len(token_ids) > self.context_length:
            raise RuntimeError(
                "prompt is too long for CLIP context length %d: %s"
                % (self.context_length, template.format(label))
            )
        label_start = 1 + len(prefix_ids)
        label_end = label_start + len(label_ids)
        return token_ids, (label_start, label_end), len(token_ids) - 1
