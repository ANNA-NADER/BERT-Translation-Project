from src.utils.tokenizer import get_tokenizers

source, target = get_tokenizers()

print(f"Source: {source.name_or_path}, Vocab: {len(source)}")
print(f"Target: {target.name_or_path}, Vocab: {len(target)}")

assert len(source) == 30522
assert len(target) == 30522
print("SUCCESS: Vocabularies match BERT base!")
