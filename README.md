# Simple Chat App with Semantic Kernel

A command-line chat app that uses Microsoft's [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
to talk to OpenAI or Azure OpenAI, keep conversation history, and call plugins automatically.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Builds the kernel, runs the chat loop, handles slash-commands |
| `plugins.py` | `TimePlugin` and `MathPlugin` the model can call as tools |
| `requirements.txt` | Python dependencies |
| `.env.example` | Template for your API settings |

## Setup

1. Install Python 3.10 or newer.
2. Put all the files in one folder, then create a virtual environment and install dependencies:

   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and fill in your key. Pick `openai` or `azure` with `AI_PROVIDER`.

4. Run it:

   ```bash
   python main.py
   ```

## Using it

Just type messages. Try:

- `What time is it?` (calls `time.current_datetime`)
- `What is (1234 * 56) / 7?` (calls `math.calculate`)
- `My name is Sam.` followed by `What's my name?` (history is remembered)

Commands: `/help`, `/history`, `/reset`, `/save`, `/exit`.

## How it works

1. **Kernel**: `Kernel()` is the central object. An AI service and plugins are registered on it.
2. **Service**: `OpenAIChatCompletion` or `AzureChatCompletion` sends the conversation to the model.
3. **ChatHistory**: every user and assistant message is appended, so each call includes prior context.
4. **Plugins**: methods decorated with `@kernel_function` become tools. `FunctionChoiceBehavior.Auto()`
   lets the model decide when to call them and feeds the results back automatically.

## Extending it

- Add a new plugin class in `plugins.py` and register it in `build_kernel()` with `kernel.add_plugin(...)`.
- Switch to streaming output with `chat_service.get_streaming_chat_message_content(...)`.
- Change the personality by editing `SYSTEM_PROMPT` in `main.py`.

## Troubleshooting

- **Missing environment variables**: make sure the file is named `.env` (not `.env.example`) and sits next to `main.py`.
- **Import errors**: run `pip install -U semantic-kernel`. Semantic Kernel's Python API changes between releases.
- **401 or 404 from Azure**: `AZURE_OPENAI_DEPLOYMENT` must be your *deployment name*, not the model name.
