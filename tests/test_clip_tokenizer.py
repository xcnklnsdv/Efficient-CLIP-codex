import torch

from models.clip_tokenizer import OpenAIClipBPETokenizer
from models.emclip import PromptTextEncoder


def test_official_clip_bpe_ids_and_label_span():
    tokenizer = OpenAIClipBPETokenizer(context_length=77)

    token_ids, label_span, eot_position = tokenizer.encode_prompt("a photo of a {}", "dog")

    assert token_ids == [49406, 320, 1125, 539, 320, 1929, 49407]
    assert label_span == (5, 6)
    assert eot_position == 6
    assert all(token not in (tokenizer.pad_id, tokenizer.sot_id, tokenizer.eot_id)
               for token in token_ids[label_span[0]:label_span[1]])


def test_text_eot_cannot_attend_to_padding_tokens():
    torch.manual_seed(4)
    encoder = PromptTextEncoder(
        ["dog"],
        width=16,
        layers=1,
        heads=4,
        embed_dim=8,
        context_length=12,
    )
    tokens, _, eot_positions, _, _ = encoder._tokenize_templates(torch.device("cpu"))
    before = encoder._encode_tokens(tokens)[0, eot_positions[0]].detach().clone()
    with torch.no_grad():
        encoder.token_embedding.weight[encoder.tokenizer.pad_id].add_(1000.0)
    after = encoder._encode_tokens(tokens)[0, eot_positions[0]].detach()

    assert torch.allclose(before, after, atol=1e-6, rtol=1e-6)


def test_training_transition_invalidates_text_evaluation_cache():
    torch.manual_seed(9)
    encoder = PromptTextEncoder(
        ["running", "jumping"],
        width=16,
        layers=1,
        heads=4,
        embed_dim=8,
        context_length=12,
    )
    encoder.eval()
    cached = encoder.encode_class_prompts(training_mode=False)[0].clone()
    encoder.train()
    with torch.no_grad():
        encoder.token_embedding.weight.add_(torch.randn_like(encoder.token_embedding.weight) * 0.1)
    encoder.eval()
    refreshed = encoder.encode_class_prompts(training_mode=False)[0]

    assert not torch.allclose(cached, refreshed)
