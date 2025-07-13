#!/usr/bin/env python3
import re

# Sample LLM response from the log
sample_response = """[CONTINUES]False[/CONTINUES]
[HEAD]Do Large Language Models Need a Content Delivery Network? > Abstract > Abstract[/HEAD]
As the use of large language models (LLMs) expands rapidly, so does the range of knowledge needed to supplement various LLM queries. Thus, enabling modular and efficient injection of new knowledge in LLM inference is critical.

[CONTINUES]False[/CONTINUES]
[HEAD]Do Large Language Models Need a Content Delivery Network? > 1 Background and Motivation > 1 Background and Motivation[/HEAD]
Traditionally, machine learning models, such as computer vision [22, 30, 31, 38] and image generation [20, 32, 37], learn all the knowledge from the training data."""

# Test the regex pattern
chunk_pattern = re.compile(
    r'\[CONTINUES\](True|False|Partial)\[/CONTINUES\]\s*\[HEAD\](.*?)\[/HEAD\]\s*(.*?)(?=\s*\[CONTINUES\]|\Z)',
    re.DOTALL | re.MULTILINE
)

matches = chunk_pattern.findall(sample_response)

print(f"Found {len(matches)} matches")
for i, match in enumerate(matches):
    continues_flag, heading, content = match
    print(f"\nChunk {i+1}:")
    print(f"  Continues: {continues_flag}")
    print(f"  Heading: {heading}")
    print(f"  Content preview: {content[:100]}...")
