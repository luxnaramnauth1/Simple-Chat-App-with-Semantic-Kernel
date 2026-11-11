"""A simple chat app built with Semantic Kernel.

Features:
  * OpenAI or Azure OpenAI chat completion (set in .env)
  * Persistent chat history within a session (the model remembers context)
  * Automatic function calling with two sample plugins
  * Small slash-commands: /help, /reset, /history, /save, /exit
"""

import asyncio
import json
import os
import sys

from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai import (
    AzureChatCompletion,
    OpenAIChatCompletion,
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.contents import ChatHistory

from plugins import MathPlugin, TimePlugin

SYSTEM_PROMPT = (
    "You are a friendly, concise assistant. Use the available tools when they help, "
    "for example for the current time or for arithmetic. If you don't know something, say so."
)

SERVICE_ID = "chat"


def build_kernel() -> Kernel:
    """Create the kernel, attach an AI service, and register plugins."""
    load_dotenv()
    provider = os.getenv("AI_PROVIDER", "openai").strip().lower()
    kernel = Kernel()

    if provider == "azure":
        required = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_DEPLOYMENT"]
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            sys.exit(f"Missing environment variables in .env: {', '.join(missing)}")
        service = AzureChatCompletion(
            service_id=SERVICE_ID,
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        )
    elif provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit("Missing OPENAI_API_KEY in .env (copy .env.example to .env first).")
        service = OpenAIChatCompletion(
            service_id=SERVICE_ID,
            api_key=os.environ["OPENAI_API_KEY"],
            ai_model_id=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        )
    else:
        sys.exit(f"Unknown AI_PROVIDER '{provider}'. Use 'openai' or 'azure'.")

    kernel.add_service(service)
    kernel.add_plugin(TimePlugin(), plugin_name="time")
    kernel.add_plugin(MathPlugin(), plugin_name="math")
    return kernel


def new_history() -> ChatHistory:
    history = ChatHistory()
    history.add_system_message(SYSTEM_PROMPT)
    return history


def save_history(history: ChatHistory, path: str = "chat_history.json") -> None:
    data = [
        {"role": str(m.role.value), "content": str(m.content)}
        for m in history.messages
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} messages to {path}")


def print_help() -> None:
    print(
        "Commands:\n"
        "  /help     show this help\n"
        "  /history  show the conversation so far\n"
        "  /reset    clear the conversation\n"
        "  /save     write the conversation to chat_history.json\n"
        "  /exit     quit"
    )


async def main() -> None:
    kernel = build_kernel()
    chat_service = kernel.get_service(SERVICE_ID)

    settings = OpenAIChatPromptExecutionSettings(
        service_id=SERVICE_ID,
        temperature=0.7,
        max_tokens=800,
        function_choice_behavior=FunctionChoiceBehavior.Auto(),
    )

    history = new_history()
    print("Semantic Kernel chat. Type /help for commands.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            command = user_input.lower()
            if command == "/exit":
                print("Goodbye!")
                break
            if command == "/help":
                print_help()
            elif command == "/reset":
                history = new_history()
                print("Conversation cleared.")
            elif command == "/history":
                for m in history.messages:
                    print(f"[{m.role.value}] {m.content}")
            elif command == "/save":
                save_history(history)
            else:
                print("Unknown command. Type /help.")
            continue

        history.add_user_message(user_input)
        try:
            reply = await chat_service.get_chat_message_content(
                chat_history=history,
                settings=settings,
                kernel=kernel,
            )
        except Exception as exc:  # noqa: BLE001
            # Drop the failed user turn so history stays consistent.
            history.messages.pop()
            print(f"Error talking to the model: {exc}\n")
            continue

        history.add_message(reply)
        print(f"Bot: {reply.content}\n")


if __name__ == "__main__":
    asyncio.run(main())
