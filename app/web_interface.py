"""
Bidirectional English <-> French web interface using Helsinki-NLP MarianMT models.
Compatible with transformers v5.x.
"""

import sys
from pathlib import Path

import gradio as gr
import torch
from transformers import MarianMTModel, MarianTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))


MODELS = {
    "en-fr": "Helsinki-NLP/opus-mt-en-fr",
    "fr-en": "Helsinki-NLP/opus-mt-fr-en",
}

LABELS = {
    "en-fr": ("English", "French"),
    "fr-en": ("French", "English"),
}


class Translator:
    """Wrapper around a MarianMT model for one translation direction."""

    def __init__(self, model_name: str):
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model = MarianMTModel.from_pretrained(model_name)
        self.model.eval()

    def translate(
        self,
        text: str,
        num_beams: int = 5,
        max_length: int = 256,
    ) -> str:
        inputs = self.tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                num_beams=num_beams,
                max_length=max_length,
            )
        return self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
        )


class TranslationInterface:
    """Gradio bidirectional translation interface using Helsinki-NLP MarianMT."""

    def __init__(self):
        print("Loading EN -> FR model...")
        self.translators = {
            "en-fr": Translator(MODELS["en-fr"]),
        }
        print("Loading FR -> EN model...")
        self.translators["fr-en"] = Translator(MODELS["fr-en"])
        print("Both models ready.")

    def translate(self, text: str, direction: str, num_beams: int, max_length: int) -> str:
        """Run translation in the selected direction."""
        if not text.strip():
            return ""
        try:
            return self.translators[direction].translate(
                text,
                num_beams=int(num_beams),
                max_length=int(max_length),
            )
        except Exception as e:
            return f"Error: {e}"

    def create_interface(self):
        """Build the Gradio UI."""

        css = """
        body { font-family: 'Inter', sans-serif; }
        #swap-btn { font-size: 1.4rem; min-width: 48px; }
        footer { display: none !important; }
        """

        with gr.Blocks(css=css, title="EN ↔ FR Translator") as interface:
            gr.Markdown(
                """# English ↔ French Translator
Powered by [Helsinki-NLP MarianMT](https://huggingface.co/Helsinki-NLP)
— production-quality neural machine translation.
                """
            )

            direction = gr.State("en-fr")

            with gr.Row():
                src_label = gr.Markdown("**English**")
                swap_btn = gr.Button("⇄", elem_id="swap-btn", scale=0)
                tgt_label = gr.Markdown("**French**")

            with gr.Row():
                input_box = gr.Textbox(
                    placeholder="Enter text to translate...",
                    lines=6,
                    label="",
                    show_label=False,
                )
                output_box = gr.Textbox(
                    lines=6,
                    label="",
                    show_label=False,
                    interactive=False,
                    placeholder="Translation will appear here...",
                )

            with gr.Row():
                translate_btn = gr.Button("Translate", variant="primary", scale=2)
                clear_btn = gr.Button("Clear", scale=1)

            with gr.Accordion("Options", open=False):
                num_beams = gr.Slider(1, 10, value=5, step=1, label="Beam Size")
                max_length = gr.Slider(32, 512, value=256, step=16, label="Max Output Length")

            gr.Markdown("### Examples")
            gr.Examples(
                examples=[
                    ["Hello, how are you today?", "en-fr"],
                    ["I like potatoes.", "en-fr"],
                    ["Paris est la capitale de la France.", "fr-en"],
                    ["J'aime apprendre de nouvelles langues.", "fr-en"],
                    ["Le temps est magnifique ce matin.", "fr-en"],
                ],
                inputs=[input_box, direction],
                label="Click an example to load it",
            )

            with gr.Accordion("Model Information", open=False):
                gr.Markdown("""
| Direction | Model | Parameters |
|-----------|-------|------------|
| EN → FR | `Helsinki-NLP/opus-mt-en-fr` | ~74M |
| FR → EN | `Helsinki-NLP/opus-mt-fr-en` | ~74M |

Both models use the [MarianMT](https://huggingface.co/docs/transformers/model_doc/marian)
architecture, trained on the OPUS corpus (hundreds of millions of sentence pairs).
                """)

            def do_translate(text, direc, beams, maxlen):
                return self.translate(text, direc, beams, maxlen)

            def do_swap(direc, inp, out):
                new_dir = "fr-en" if direc == "en-fr" else "en-fr"
                src, tgt = LABELS[new_dir]
                return new_dir, out, inp, f"**{src}**", f"**{tgt}**"

            translate_btn.click(
                fn=do_translate,
                inputs=[input_box, direction, num_beams, max_length],
                outputs=output_box,
            )

            input_box.submit(
                fn=do_translate,
                inputs=[input_box, direction, num_beams, max_length],
                outputs=output_box,
            )

            clear_btn.click(
                fn=lambda: ("", ""),
                inputs=None,
                outputs=[input_box, output_box],
            )

            swap_btn.click(
                fn=do_swap,
                inputs=[direction, input_box, output_box],
                outputs=[direction, input_box, output_box, src_label, tgt_label],
            )

        return interface

    def launch(self, share: bool = False, server_port: int = 7860):
        interface = self.create_interface()
        interface.launch(share=share, server_port=server_port)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="EN <-> FR Translation Web Interface")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    args = parser.parse_args()

    app = TranslationInterface()
    app.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
