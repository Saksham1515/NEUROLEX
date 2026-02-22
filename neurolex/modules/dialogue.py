"""
Module 08 — Conversational Dialogue System
DialoGPT-medium for multi-turn conversation with session memory.
"""
from __future__ import annotations
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from neurolex.config import MODELS, DIALOGUE_CONFIG, DEVICE


@st.cache_resource(show_spinner=False)
def _load_dialogue_model():
    cfg = MODELS["dialogue"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"])
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


class DialogueSystem:
    """
    Multi-turn conversational dialogue using DialoGPT-medium.

    Architecture:
        - Model: microsoft/DialoGPT-medium (117M params)
        - Approach: Autoregressive GPT-2 fine-tuned on Reddit conversations
        - Context: Concatenated conversation history encoded as single sequence
        - Memory: In-session history kept as list of token tensors

    Context Management:
        - Maximum context window: 1024 tokens
        - History is truncated FIFO when exceeding limit
        - Each turn: [prev_responses + EOS + new_input]

    Evaluation:
        - Perplexity
        - BLEU-4 on held-out dialogues
        - Human eval: coherence, engagement, safety

    Extensions:
        - Persona conditioning (PersonaChat)
        - Knowledge-grounded dialogue (Wizard of Wikipedia)
        - Safety layer with toxicity filtering
    """

    MAX_HISTORY_TOKENS = 900

    def __init__(self):
        self.tokenizer, self.model = _load_dialogue_model()
        self.history_ids: list[torch.Tensor] = []

    def chat(self, user_input: str) -> str:
        """
        Generate a response to user input, maintaining conversation history.

        Args:
            user_input: The user's message

        Returns:
            str: Model's response
        """
        # Encode user input + EOS
        new_input_ids = self.tokenizer.encode(
            user_input + self.tokenizer.eos_token, return_tensors="pt"
        ).to(DEVICE)

        # Concatenate with history
        if self.history_ids:
            bot_input_ids = torch.cat(self.history_ids + [new_input_ids], dim=-1)
        else:
            bot_input_ids = new_input_ids

        # Truncate if too long
        if bot_input_ids.shape[-1] > self.MAX_HISTORY_TOKENS:
            bot_input_ids = bot_input_ids[:, -self.MAX_HISTORY_TOKENS:]

        with torch.no_grad():
            output_ids = self.model.generate(
                bot_input_ids,
                max_new_tokens=DIALOGUE_CONFIG["max_new_tokens"],
                do_sample=DIALOGUE_CONFIG["do_sample"],
                top_p=DIALOGUE_CONFIG["top_p"],
                temperature=DIALOGUE_CONFIG["temperature"],
                pad_token_id=DIALOGUE_CONFIG["pad_token_id"],
            )

        # Extract only the new tokens
        response_ids = output_ids[:, bot_input_ids.shape[-1]:]
        response = self.tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Update history
        self.history_ids.append(new_input_ids)
        self.history_ids.append(output_ids[:, bot_input_ids.shape[-1]:])

        # Keep history bounded
        total_tokens = sum(h.shape[-1] for h in self.history_ids)
        while total_tokens > self.MAX_HISTORY_TOKENS and len(self.history_ids) > 2:
            removed = self.history_ids.pop(0)
            total_tokens -= removed.shape[-1]

        return response.strip() if response.strip() else "I'm not sure how to respond to that."

    def reset(self):
        """Clear conversation history."""
        self.history_ids = []

    def get_history_length(self) -> int:
        """Return total number of history tokens."""
        return sum(h.shape[-1] for h in self.history_ids)
