# How to read the TUI

The top panel shows the original prompt, the text generated so far, and the latest token. The token number is its tokenizer ID. Press `q` or `Esc` to stop generation.

Each attention panel is labeled with a layer and head, such as `L23 H4`. Rows represent query tokens and columns represent tokens in the context. Brighter cells mean the model placed more attention on that position. During prompt processing, the map has several rows. During KV-cached generation, only one new token is processed, so the map is usually one row across the full context. If the context is wider than the panel, the TUI shows the most recent positions.

The candidate logits table shows the tokens the model considered most likely. A larger logit means a stronger preference, but logits are raw scores rather than percentages. The highest-scoring token is selected for the current greedy decoding flow.

The Session card shows prediction steps, emitted tokens, current context size, and total inference time. Context is the number of tokens currently visible through the prompt and KV cache.

The Latency card shows the latest token time, average token time, and time to first token. The first token is normally slower because it processes the entire prompt. Later tokens are faster because they reuse cached keys and values.

The Throughput card reports generated tokens per second. “Overall” covers the whole run, while “Latest 5” better reflects current decoding speed after prompt processing.

The Distribution card summarizes token latency. `P50` is the median, `P95` represents slower tokens near the upper end, and Minimum and Maximum show the observed range.

The tensor-flow table shows operations performed by each transformer layer. Shapes are generally written as `[batch, sequence length, feature size]`. During cached generation, `[1, 1, 1536]` is normal: one batch, one new token, and 1,536 hidden features. Earlier context is stored in the KV cache rather than repeated in this shape.

Operations such as normalization and residual addition preserve the hidden-state shape. SwiGLU expands the features internally and projects them back, so its displayed input and output can both be `[1, 1, 1536]`. The LM head changes the final dimension to the vocabulary size, such as `[1, 1, 130560]`, producing one score for every possible token.

The time column shows how long each operation took. `0.00 ms` usually means the operation was faster than the display precision, not that no work occurred.
