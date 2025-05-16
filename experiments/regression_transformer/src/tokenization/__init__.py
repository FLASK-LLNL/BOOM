from .rt_tokenizers import ExpressionBertTokenizer
from arguments import Arguments


def setup_tokenizer(args: Arguments) -> ExpressionBertTokenizer:
    return ExpressionBertTokenizer.from_pretrained(args.tokenizer)
