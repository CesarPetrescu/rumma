# Rumma Interactive Chat Mode

## Overview

The Rumma CLI now supports an interactive chat mode that allows you to:
- Load models from HuggingFace
- See download progress
- Enter prompts and tokenize input
- Understand the inference pipeline

## Usage

### Running in Interactive Mode

```bash
cargo run -p rumma-cli -- --model https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ --interactive
```

### What You'll See

1. **Download Progress**: Clear indicators when downloading model and tokenizer
   ```
   📦 Downloading from HuggingFace: Qwen/Qwen2.5-3B-Instruct-AWQ/model.safetensors
      Revision: main
      Fetching model file...
   ✓ Download complete
   ```

2. **Interactive Prompt**: A friendly chat interface
   ```
   ═══════════════════════════════════════════════════════════
     Welcome to Rumma Interactive Chat!
   ═══════════════════════════════════════════════════════════

   Type your message and press Enter to chat.
   Type 'quit' or 'exit' to leave.
   ```

3. **Tokenization Debug Info**: See how your text is tokenized
   ```
   [DEBUG] Tokenized input:
     Tokens: [123, 456, 789, ...]
     Token count: 15
   ```

### Example Session

```bash
$ cargo run -p rumma-cli -- --model https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-AWQ --interactive

📦 Downloading from HuggingFace: Qwen/Qwen2.5-3B-Instruct-AWQ/model.safetensors
   Revision: main
   Fetching model file...
✓ Download complete

Loaded model: hidden_size=2048 layers=36 source=...

🤖 Entering interactive chat mode...

📝 Downloading tokenizer from HuggingFace: Qwen/Qwen2.5-3B-Instruct-AWQ
✓ Tokenizer download complete

═══════════════════════════════════════════════════════════
  Welcome to Rumma Interactive Chat!
═══════════════════════════════════════════════════════════

Type your message and press Enter to chat.
Type 'quit' or 'exit' to leave.

Model: hidden_size=2048 layers=36
═══════════════════════════════════════════════════════════

You: Hello, how are you?

[DEBUG] Tokenized input:
  Tokens: [9456, 11, 1268, 527, 499, 30]
  Token count: 6