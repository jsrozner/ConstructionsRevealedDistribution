from lib.common.mlm_singleton import get_singleton_scorer

def count_params(**kwargs):
    mlm = get_singleton_scorer()
    total_params = sum(p.numel() for p in mlm.model.parameters())
    vocab_size = mlm.tokenizer.vocab_size
    hidden_size = mlm.model.config.hidden_size

    params_no_vocab = total_params - vocab_size * hidden_size

    return total_params, params_no_vocab, vocab_size
